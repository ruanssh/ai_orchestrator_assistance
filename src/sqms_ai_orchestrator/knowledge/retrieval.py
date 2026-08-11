from collections import defaultdict

import bm25s

from .documents import DocumentSection, RankedSection


class LexicalRetriever:
    def __init__(self) -> None:
        self.sections: list[DocumentSection] = []
        self.retriever: bm25s.BM25 | None = None

    def index(self, sections: list[DocumentSection]) -> None:
        self.sections = sections
        if not sections:
            self.retriever = None
            return
        corpus = [{"index": index} for index in range(len(sections))]
        tokens = bm25s.tokenize([section.search_text for section in sections], stopwords="pt")
        self.retriever = bm25s.BM25(corpus=corpus)
        self.retriever.index(tokens, show_progress=False)

    def search(self, query: str, limit: int) -> list[RankedSection]:
        if self.retriever is None or not self.sections:
            return []
        query_tokens = bm25s.tokenize([query], stopwords="pt")
        results, scores = self.retriever.retrieve(
            query_tokens,
            k=min(limit, len(self.sections)),
            show_progress=False,
        )
        ranked: list[RankedSection] = []
        for result, score in zip(results[0], scores[0], strict=True):
            index = int(result["index"])
            value = float(score)
            if value <= 0:
                continue
            ranked.append(RankedSection(self.sections[index], value, lexical_score=value))
        return ranked


class SemanticRetriever:
    def __init__(self, model_name: str, enabled: bool):
        self.model_name = model_name
        self.enabled = enabled
        self.model = None
        self.embeddings = None
        self.sections: list[DocumentSection] = []

    def index(self, sections: list[DocumentSection]) -> None:
        self.sections = sections
        if not self.enabled or not sections:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Instale o extra 'semantic' para habilitar embeddings.") from error
        self.model = SentenceTransformer(self.model_name)
        self.embeddings = self.model.encode(
            [section.search_text for section in sections],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, limit: int) -> list[RankedSection]:
        if not self.enabled or self.model is None or self.embeddings is None:
            return []
        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ query_embedding
        indices = scores.argsort()[::-1][:limit]
        return [
            RankedSection(self.sections[int(index)], float(scores[index]), semantic_score=float(scores[index]))
            for index in indices
            if float(scores[index]) > 0
        ]


def reciprocal_rank_fusion(result_sets: list[list[RankedSection]], limit: int, k: int = 60) -> list[RankedSection]:
    fused: dict[str, float] = defaultdict(float)
    entries: dict[str, RankedSection] = {}
    for results in result_sets:
        for rank, result in enumerate(results, start=1):
            fused[result.section.id] += 1 / (k + rank)
            existing = entries.get(result.section.id)
            if existing is None:
                entries[result.section.id] = result
            else:
                existing.lexical_score = max(existing.lexical_score, result.lexical_score)
                existing.semantic_score = max(existing.semantic_score, result.semantic_score)
    ordered = sorted(fused, key=fused.get, reverse=True)[:limit]
    for section_id in ordered:
        entries[section_id].score = fused[section_id]
    return [entries[section_id] for section_id in ordered]


class OptionalReranker:
    def __init__(self, model_name: str, enabled: bool):
        self.model_name = model_name
        self.enabled = enabled
        self.model = None

    def rerank(self, query: str, results: list[RankedSection], limit: int) -> list[RankedSection]:
        if not self.enabled or not results:
            return results[:limit]
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError("Instale o extra 'semantic' para habilitar reranking.") from error
            self.model = CrossEncoder(self.model_name)
        scores = self.model.predict([(query, result.section.search_text) for result in results])
        rescored = sorted(zip(results, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
        for result, score in rescored:
            result.score = float(score)
        return [result for result, _ in rescored[:limit]]
