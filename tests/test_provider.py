import json

import pytest

from sqms_ai_orchestrator.models import ChatMessage
from sqms_ai_orchestrator.provider import (
    OpenAICompatibleProvider,
    _REASONING_CACHE,
    extract_json_object,
)


def test_extract_json_from_model_markdown() -> None:
    assert extract_json_object('```json\n{"queries":["cotação"]}\n```') == {"queries": ["cotação"]}


@pytest.mark.asyncio
async def test_provider_does_not_limit_output_tokens() -> None:
    class FakeResponse:
        status_code = 200
        is_success = True

        @staticmethod
        def json():
            return {'choices': [{'message': {'content': 'Resposta completa'}}]}

    class FakeClient:
        def __init__(self):
            self.request = None

        async def post(self, url, **kwargs):
            self.request = (url, kwargs)
            return FakeResponse()

    provider = OpenAICompatibleProvider('http://gateway.test/v1', 'EMPTY', 'auto', 120)
    client = FakeClient()
    provider.client = client

    result = await provider.complete([ChatMessage(role='user', content='Pergunta')])

    assert result == 'Resposta completa'
    assert 'max_tokens' not in client.request[1]['json']


@pytest.mark.asyncio
async def test_provider_negotiates_and_caches_reasoning_effort(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.is_success = status_code == 200
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self):
            self.requests = []
            self.responses = [
                FakeResponse(400, {
                    'error': {
                        'message': "'reasoning_effort' must be one of: 'low', 'medium', 'high'",
                    },
                }),
                FakeResponse(200, {'choices': [{'message': {'content': 'Resposta'}}]}),
            ]

        async def post(self, url, **kwargs):
            self.requests.append(kwargs)
            return self.responses.pop(0)

    monkeypatch.delenv('SQMS_AI_REASONING_EFFORT', raising=False)
    cache_key = ('http://reasoning.test/v1', 'auto')
    _REASONING_CACHE.pop(cache_key, None)
    provider = OpenAICompatibleProvider(cache_key[0], 'EMPTY', cache_key[1], 120)
    client = FakeClient()
    provider.client = client

    await provider.complete([ChatMessage(role='user', content='Pergunta')], thinking=False)

    assert client.requests[0]['json']['reasoning_effort'] == 'none'
    assert client.requests[1]['json']['reasoning_effort'] == 'low'
    assert _REASONING_CACHE[cache_key] == 'low'
    _REASONING_CACHE.pop(cache_key, None)


@pytest.mark.asyncio
async def test_provider_omits_unsupported_reasoning_parameter(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        is_success = True
        text = json.dumps({'choices': [{'message': {'content': 'Resposta'}}]})

        @staticmethod
        def json():
            return {'choices': [{'message': {'content': 'Resposta'}}]}

    class FakeClient:
        def __init__(self):
            self.requests = []

        async def post(self, url, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return type('Rejected', (), {
                    'status_code': 400,
                    'is_success': False,
                    'text': json.dumps({'error': {'message': 'Unknown parameter reasoning_effort'}}),
                })()
            return FakeResponse()

    monkeypatch.setenv('SQMS_AI_REASONING_EFFORT', 'custom')
    cache_key = ('http://omit.test/v1', 'auto')
    _REASONING_CACHE.pop(cache_key, None)
    provider = OpenAICompatibleProvider(cache_key[0], 'EMPTY', cache_key[1], 120)
    client = FakeClient()
    provider.client = client

    await provider.complete([ChatMessage(role='user', content='Pergunta')], thinking=False)

    assert client.requests[0]['json']['reasoning_effort'] == 'custom'
    assert 'reasoning_effort' not in client.requests[1]['json']
    assert _REASONING_CACHE[cache_key] is None
    _REASONING_CACHE.pop(cache_key, None)
