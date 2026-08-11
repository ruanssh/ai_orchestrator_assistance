import pytest

from sqms_ai_orchestrator.models import ChatMessage
from sqms_ai_orchestrator.provider import OpenAICompatibleProvider, extract_json_object


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
