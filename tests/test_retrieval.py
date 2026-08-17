from pathlib import Path

from sqms_ai_orchestrator.knowledge.documents import DocumentSection
from sqms_ai_orchestrator.knowledge.retrieval import LexicalRetriever


def section(identifier: str, heading: str, content: str) -> DocumentSection:
    return DocumentSection(
        id=identifier,
        document_id="test",
        source=Path("test.md"),
        document_title="Teste",
        heading=heading,
        heading_path=(heading,),
        level=1,
        content=content,
        start_line=1,
        end_line=2,
    )


def test_lexical_retrieval_finds_operational_section() -> None:
    retriever = LexicalRetriever()
    retriever.index([
        section("create", "Como criar", "Acesse Aprovações e clique em Nova solicitação."),
        section("deadline", "Prazo", "O prazo médio da cotação é de sete dias úteis."),
    ])
    results = retriever.search("como criar nova solicitação de cotação", 2)
    assert results
    assert results[0].section.id == "create"


def test_real_procedures_return_expected_documents() -> None:
    from sqms_ai_orchestrator.config import get_settings
    from sqms_ai_orchestrator.flows import FlowRegistry
    from sqms_ai_orchestrator.knowledge import KnowledgeBase

    settings = get_settings()
    flows = FlowRegistry(settings.flows_directory)
    flows.load()
    knowledge = KnowledgeBase(
        semantic_enabled=False,
        embedding_model=settings.embedding_model,
        rerank_enabled=False,
        rerank_model=settings.rerank_model,
        lexical_candidates=20,
        semantic_candidates=15,
        final_candidates=8,
    )
    counts = knowledge.reindex(flows.all())
    assert counts["procurement"] > 100

    cases = {
        "como criar uma cotação no SQMS": "RAG_SQMS_CAMPOS.md",
        "como adicionar aprovadores": "RAG_SQMS_CAMPOS.md",
        "quem aprova pedido de compra acima de R$ 10.000": "RAG_COTAÇÃO_COMPRAS.md",
        "notebook para AutoCAD": "RAG_BASELINE_NOTEBOOK_COMPUTADOR.md",
        "quem é o GM": "RAG_SQMS_Organizacional.md",
        "quem é nadyson": "RAG_SQMS_Organizacional.md",
        "quem é celio": "RAG_SQMS_Organizacional.md",
        "gerente de T.I.": "RAG_SQMS_Organizacional.md",
    }
    for query, expected_document in cases.items():
        results = knowledge.search(query, "procurement", 5)
        assert results
        assert results[0].section.source.name == expected_document

    organizational = knowledge.search(
        'gerente de TI',
        'procurement',
        5,
        source_name='RAG_SQMS_Organizacional.md',
    )
    assert organizational
    assert organizational[0].section.heading_path[-1] == 'Nadyson Oliveira'

    celio = knowledge.search(
        'quem é celio',
        'procurement',
        5,
        source_name='RAG_SQMS_Organizacional.md',
    )
    assert celio
    assert celio[0].section.heading_path[-1] == 'Célio Oliveira'
