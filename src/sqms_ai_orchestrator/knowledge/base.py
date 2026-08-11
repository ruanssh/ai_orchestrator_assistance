from pathlib import Path

from ..flows import FlowConfig
from .documents import DocumentSection, RankedSection
from .markdown_parser import MarkdownSectionParser
from .retrieval import LexicalRetriever, OptionalReranker, SemanticRetriever, reciprocal_rank_fusion


class KnowledgeBase:
    def __init__(
        self,
        *,
        semantic_enabled: bool,
        embedding_model: str,
        rerank_enabled: bool,
        rerank_model: str,
        lexical_candidates: int,
        semantic_candidates: int,
        final_candidates: int,
    ):
        self.parser = MarkdownSectionParser()
        self.lexical = LexicalRetriever()
        self.semantic = SemanticRetriever(embedding_model, semantic_enabled)
        self.reranker = OptionalReranker(rerank_model, rerank_enabled)
        self.lexical_candidates = lexical_candidates
        self.semantic_candidates = semantic_candidates
        self.final_candidates = final_candidates
        self.sections: dict[str, DocumentSection] = {}
        self.flow_sections: dict[str, set[str]] = {}

    def reindex(self, flows: list[FlowConfig]) -> dict[str, int]:
        all_sections: dict[str, DocumentSection] = {}
        flow_sections: dict[str, set[str]] = {}
        for flow in flows:
            ids: set[str] = set()
            for knowledge_path in flow.knowledge_paths:
                paths = sorted(knowledge_path.glob("**/*.md")) if knowledge_path.is_dir() else [knowledge_path]
                for path in paths:
                    if not path.is_file():
                        continue
                    for section in self.parser.parse_file(path):
                        all_sections[section.id] = section
                        ids.add(section.id)
            flow_sections[flow.id] = ids
        self.sections = all_sections
        self.flow_sections = flow_sections
        section_list = list(all_sections.values())
        self.lexical.index(section_list)
        self.semantic.index(section_list)
        return {flow_id: len(ids) for flow_id, ids in flow_sections.items()}

    def search(self, query: str, flow_id: str, limit: int | None = None) -> list[RankedSection]:
        allowed = self.flow_sections.get(flow_id, set())
        lexical = [item for item in self.lexical.search(query, self.lexical_candidates) if item.section.id in allowed]
        semantic = [item for item in self.semantic.search(query, self.semantic_candidates) if item.section.id in allowed]
        candidates = reciprocal_rank_fusion([lexical, semantic], max(self.final_candidates * 2, 12))
        return self.reranker.rerank(query, candidates, limit or self.final_candidates)

    def expand(self, selected_ids: list[str], allowed_ids: set[str]) -> list[DocumentSection]:
        selected: list[DocumentSection] = []
        added: set[str] = set()
        sections_by_source: dict[Path, list[DocumentSection]] = {}
        for section in self.sections.values():
            sections_by_source.setdefault(section.source, []).append(section)
        for values in sections_by_source.values():
            values.sort(key=lambda item: item.start_line)

        def add(section: DocumentSection | None) -> None:
            if section and section.id in allowed_ids and section.id not in added:
                added.add(section.id)
                selected.append(section)

        for section_id in selected_ids:
            section = self.sections.get(section_id)
            add(section)
            if section is None:
                continue
            add(self.sections.get(section.parent_id or ""))
            siblings = sections_by_source.get(section.source, [])
            index = next((i for i, item in enumerate(siblings) if item.id == section.id), -1)
            if index >= 0 and index + 1 < len(siblings):
                next_section = siblings[index + 1]
                if next_section.parent_id == section.parent_id:
                    add(next_section)
            for child in siblings:
                if child.parent_id == section.id:
                    add(child)
        return selected
