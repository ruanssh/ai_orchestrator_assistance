import hashlib
import re
from pathlib import Path

import frontmatter
from markdown_it import MarkdownIt

from .documents import DocumentSection


SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    normalized = value.lower().encode("ascii", "ignore").decode("ascii")
    return SLUG_PATTERN.sub("-", normalized).strip("-") or "section"


class MarkdownSectionParser:
    def __init__(self) -> None:
        self.parser = MarkdownIt("commonmark")

    def parse_file(self, path: Path) -> list[DocumentSection]:
        post = frontmatter.load(path, encoding="utf-8-sig")
        content = post.content
        lines = content.splitlines()
        headings: list[tuple[int, int, str]] = []
        tokens = self.parser.parse(content)

        for index, token in enumerate(tokens):
            if token.type != "heading_open" or token.map is None:
                continue
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            title = inline.content.strip() if inline and inline.type == "inline" else "Seção"
            headings.append((token.map[0], int(token.tag[1]), title))

        document_title = str(post.metadata.get("titulo") or (headings[0][2] if headings else path.stem))
        document_id = str(post.metadata.get("codigo") or path.stem)
        if not headings:
            return [self._section(path, document_id, document_title, post.metadata, lines, 0, len(lines), 1, document_title, (document_title,), None)]

        sections: list[DocumentSection] = []
        stack: list[tuple[int, str, str]] = []
        ids_seen: set[str] = set()
        for position, (start, level, heading) in enumerate(headings):
            end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][2] if stack else None
            heading_path = tuple(item[1] for item in stack) + (heading,)
            base_id = f"{slugify(path.stem)}:{slugify(document_id)}:{slugify('-'.join(heading_path))}"
            section_id = base_id
            if section_id in ids_seen:
                suffix = hashlib.sha1(f"{path}:{start}".encode()).hexdigest()[:6]
                section_id = f"{base_id}-{suffix}"
            ids_seen.add(section_id)
            section = self._section(
                path, document_id, document_title, post.metadata, lines,
                start, end, level, heading, heading_path, parent_id, section_id,
            )
            sections.append(section)
            stack.append((level, heading, section.id))
        return sections

    @staticmethod
    def _section(
        path: Path,
        document_id: str,
        document_title: str,
        metadata: dict,
        lines: list[str],
        start: int,
        end: int,
        level: int,
        heading: str,
        heading_path: tuple[str, ...],
        parent_id: str | None,
        section_id: str | None = None,
    ) -> DocumentSection:
        body = "\n".join(lines[start:end]).strip()
        return DocumentSection(
            id=section_id or f"{slugify(path.stem)}:{slugify(document_id)}:{slugify(heading)}",
            document_id=document_id,
            source=path,
            document_title=document_title,
            heading=heading,
            heading_path=heading_path,
            level=level,
            content=body,
            start_line=start + 1,
            end_line=max(end, start + 1),
            metadata=dict(metadata),
            parent_id=parent_id,
        )
