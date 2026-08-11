from fastapi import APIRouter, HTTPException, Request

from .models import ChatRequest, ChatResponse, SearchRequest, SearchResponse, SearchResult


router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health(request: Request) -> dict:
    return {
        "ok": True,
        "model": request.app.state.provider.model,
        "flows": [flow.id for flow in request.app.state.flows.all()],
        "sections": len(request.app.state.knowledge.sections),
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        return await request.app.state.orchestrator.chat(
            payload.message,
            payload.session_id,
            payload.flow_id,
            payload.debug,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    flow_id = payload.flow_id or request.app.state.settings.default_flow
    try:
        request.app.state.flows.get(flow_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    results = request.app.state.knowledge.search(payload.query, flow_id, payload.limit)
    return SearchResponse(
        query=payload.query,
        results=[
            SearchResult(
                id=item.section.id,
                document=item.section.source.name,
                title=item.section.document_title,
                section=" > ".join(item.section.heading_path),
                lines=f"{item.section.start_line}-{item.section.end_line}",
                score=item.score,
                content=item.section.content,
            )
            for item in results
        ],
    )


@router.post("/knowledge/reindex")
async def reindex(request: Request) -> dict:
    counts = request.app.state.knowledge.reindex(request.app.state.flows.all())
    return {"ok": True, "sections_by_flow": counts}


@router.get("/knowledge/documents")
async def documents(request: Request) -> dict:
    grouped: dict[str, dict] = {}
    for section in request.app.state.knowledge.sections.values():
        entry = grouped.setdefault(section.source.name, {"title": section.document_title, "sections": 0})
        entry["sections"] += 1
    return {"documents": grouped}


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str, request: Request) -> dict:
    request.app.state.memory.clear(session_id)
    return {"ok": True}
