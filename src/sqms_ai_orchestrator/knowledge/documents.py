from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DocumentSection:
    id: str
    document_id: str
    source: Path
    document_title: str
    heading: str
    heading_path: tuple[str, ...]
    level: int
    content: str
    start_line: int
    end_line: int
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None

    @property
    def search_text(self) -> str:
        metadata_text = " ".join(str(value) for value in self.metadata.values())
        return f"{self.document_title}\n{' > '.join(self.heading_path)}\n{metadata_text}\n{self.content}"

    @property
    def citation_text(self) -> str:
        return (
            f"[EVIDENCE {self.id}]\n"
            f"Documento: {self.document_title} ({self.source.name})\n"
            f"Seção: {' > '.join(self.heading_path)}\n"
            f"Linhas: {self.start_line}-{self.end_line}\n"
            f"{self.content.strip()}"
        )


@dataclass(slots=True)
class RankedSection:
    section: DocumentSection
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
