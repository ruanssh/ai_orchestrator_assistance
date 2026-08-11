from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SourceReference(BaseModel):
    id: str
    document: str
    title: str
    section: str
    lines: str | None = None
    score: float | None = None


class SearchResult(SourceReference):
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    session_id: str | None = Field(default=None, min_length=8, max_length=128)
    flow_id: str | None = None
    debug: bool = False


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    flow_id: str
    model: str
    sources: list[SourceReference] = []
    debug: dict | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4_000)
    flow_id: str | None = None
    limit: int = Field(default=8, ge=1, le=30)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class SearchPlan(BaseModel):
    queries: list[str] = Field(default_factory=list, max_length=4)
    topics: list[str] = Field(default_factory=list, max_length=8)
    needs_knowledge: bool = True


class EvidenceDecision(BaseModel):
    selected_ids: list[str] = Field(default_factory=list, max_length=12)
    additional_queries: list[str] = Field(default_factory=list, max_length=3)
    ready: bool = True
