import asyncio
import json
import logging
import os
from json import JSONDecodeError
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .models import ChatMessage


T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)
_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
_REASONING_CACHE: dict[tuple[str, str], str | None] = {}
_NO_FALLBACK = object()


def _configured_reasoning_effort() -> str | None:
    value = os.getenv("SQMS_AI_REASONING_EFFORT", "auto").strip().lower()
    if value in {"", "omit", "off"}:
        return None
    return "none" if value == "auto" else value


def _reasoning_fallback(body: str, current: str | None):
    try:
        payload = json.loads(body)
        message = payload.get("error", {}).get("message", "")
    except (TypeError, JSONDecodeError):
        message = body

    normalized = str(message).lower()
    if "reasoning_effort" not in normalized:
        return _NO_FALLBACK

    supported = [value for value in _REASONING_EFFORTS if f"'{value}'" in normalized]
    for value in supported:
        if value != current:
            return value
    return None if current is not None else _NO_FALLBACK


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def close(self) -> None:
        await self.client.aclose()

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        thinking: bool = True,
        max_tokens: int | None = None,
    ) -> str:
        """
        `thinking=False` negocia o reasoning effort com o gateway. Alguns
        provedores aceitam `none`, outros aceitam apenas `low` ou não aceitam
        essa extensão; a opção válida fica em cache por URL/modelo.

        `max_tokens` só é enviado quando o raciocínio está desligado — de
        propósito. Com raciocínio ligado, o orçamento é consumido pelo próprio
        raciocínio e o conteúdo visível volta VAZIO (medido: max_tokens=250
        gastou os 250 tokens pensando e devolveu 0 caractere), o que quebraria
        o parse de JSON silenciosamente.
        """
        request_payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "temperature": temperature,
        }
        cache_key = (self.base_url, self.model)
        if not thinking:
            configured = _REASONING_CACHE.get(cache_key, _configured_reasoning_effort())
            if configured is not None:
                request_payload["reasoning_effort"] = configured
            if max_tokens is not None:
                request_payload["max_tokens"] = max_tokens

        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                compatibility_attempts = 0
                while True:
                    response = await self.client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=dict(request_payload),
                    )
                    if not thinking and response.status_code in {400, 422}:
                        current = request_payload.get("reasoning_effort")
                        fallback = _reasoning_fallback(response.text, current)
                        if fallback is not _NO_FALLBACK and compatibility_attempts < 2:
                            if fallback is None:
                                request_payload.pop("reasoning_effort", None)
                            else:
                                request_payload["reasoning_effort"] = fallback
                            _REASONING_CACHE[cache_key] = fallback
                            compatibility_attempts += 1
                            logger.info(
                                "LLM reasoning_effort fallback: %s -> %s",
                                current or "omitted",
                                fallback or "omitted",
                            )
                            continue
                    break
            except (httpx.ConnectError, httpx.ReadTimeout):
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                break
            await asyncio.sleep(0.5 * (2 ** attempt))
        if response is None:
            raise RuntimeError("O provider não respondeu.")
        if not response.is_success:
            detail = response.text[:500]
            raise RuntimeError(f"Provider respondeu {response.status_code}: {detail}")
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("O provider não retornou conteúdo.")
        return content.strip()

    async def complete_json(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> T | None:
        raw = await self.complete(messages, temperature=0.0, thinking=thinking, max_tokens=max_tokens)
        payload = extract_json_object(raw)
        if payload is None:
            return None
        try:
            return schema.model_validate(payload)
        except ValidationError:
            return None


def extract_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
