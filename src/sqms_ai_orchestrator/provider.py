import asyncio
import json
from json import JSONDecodeError
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .models import ChatMessage


T = TypeVar("T", bound=BaseModel)


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
    ) -> str:
        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [message.model_dump() for message in messages],
                        "stream": False,
                        "temperature": temperature,
                    },
                )
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

    async def complete_json(self, messages: list[ChatMessage], schema: type[T]) -> T | None:
        raw = await self.complete(messages, temperature=0.0)
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
