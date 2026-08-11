from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from .api import router
from .config import get_settings
from .flows import FlowRegistry
from .knowledge import KnowledgeBase
from .memory import ConversationMemory
from .orchestrator import AIOrchestrator
from .provider import OpenAICompatibleProvider


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_directory.mkdir(parents=True, exist_ok=True)
    flows = FlowRegistry(settings.flows_directory)
    flows.load()
    knowledge = KnowledgeBase(
        semantic_enabled=settings.semantic_enabled,
        embedding_model=settings.embedding_model,
        rerank_enabled=settings.rerank_enabled,
        rerank_model=settings.rerank_model,
        lexical_candidates=settings.lexical_candidates,
        semantic_candidates=settings.semantic_candidates,
        final_candidates=settings.final_candidates,
    )
    knowledge.reindex(flows.all())
    provider = OpenAICompatibleProvider(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_timeout_seconds,
    )
    memory = ConversationMemory(settings.data_directory / "sessions.db", settings.max_history_messages)
    app.state.settings = settings
    app.state.flows = flows
    app.state.knowledge = knowledge
    app.state.provider = provider
    app.state.memory = memory
    app.state.orchestrator = AIOrchestrator(settings, flows, knowledge, provider, memory)
    yield
    await provider.close()


app = FastAPI(title="SQMS AI Orchestrator", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


def run() -> None:
    settings = get_settings()
    uvicorn.run("sqms_ai_orchestrator.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
