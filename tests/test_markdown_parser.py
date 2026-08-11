from pathlib import Path

from sqms_ai_orchestrator.knowledge.markdown_parser import MarkdownSectionParser


def test_parser_preserves_heading_hierarchy(tmp_path: Path) -> None:
    document = tmp_path / "procedure.md"
    document.write_text(
        "---\ncodigo: TEST 001\ntitulo: Procedimento teste\n---\n\n"
        "# Processo\n\nIntrodução.\n\n## Aprovação\n\n- Gestor\n- Comprador\n",
        encoding="utf-8",
    )
    sections = MarkdownSectionParser().parse_file(document)
    approval = next(section for section in sections if section.heading == "Aprovação")
    assert approval.heading_path == ("Processo", "Aprovação")
    assert "- Gestor" in approval.content
    assert approval.parent_id is not None
