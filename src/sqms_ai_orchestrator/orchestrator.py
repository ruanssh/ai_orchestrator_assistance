import re
import uuid


_EMOJI_RE = re.compile(
    r'[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\u200D\uFE0F\U0001F3FB-\U0001F3FF]+'
)


def strip_emojis(text: str) -> str:
    """Keep assistant answers professional even when the model adds emojis."""
    cleaned = _EMOJI_RE.sub('', text)
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
    return re.sub(r' {2,}', ' ', cleaned).strip()

from .config import Settings
from .flows import FlowConfig, FlowRegistry
from .knowledge import KnowledgeBase
from .knowledge.documents import DocumentSection, RankedSection
from .memory import ConversationMemory
from .models import (
    ChatMessage,
    ChatResponse,
    EvidenceDecision,
    SearchPlan,
    SourceReference,
)
from .provider import OpenAICompatibleProvider


ORGANIZATIONAL_DOCUMENT = 'RAG_SQMS_Organizacional.md'


class AIOrchestrator:
    def __init__(
        self,
        settings: Settings,
        flows: FlowRegistry,
        knowledge: KnowledgeBase,
        provider: OpenAICompatibleProvider,
        memory: ConversationMemory,
    ):
        self.settings = settings
        self.flows = flows
        self.knowledge = knowledge
        self.provider = provider
        self.memory = memory

    async def chat(
        self,
        message: str,
        session_id: str | None,
        flow_id: str | None,
        debug: bool,
    ) -> ChatResponse:
        resolved_session = session_id or uuid.uuid4().hex
        flow = self.flows.get(flow_id or self.settings.default_flow)
        history = self.memory.get(resolved_session)
        organizational = self._is_organizational(message)
        plan = await self._plan(message, history, flow)
        # Organizational answers must always be grounded in the corporate
        # directory, even when the planner LLM fails or returns false.
        if organizational:
            plan.needs_knowledge = True
        candidates: dict[str, RankedSection] = {}
        selected_ids: list[str] = []
        queries = self._normalize_queries(plan.queries or [message], flow)
        if organizational:
            queries = [self._normalize_organizational_query(query) for query in queries]
            queries = self._normalize_queries([*queries, self._normalize_organizational_query(message)], flow)
        search_log: list[str] = []

        if plan.needs_knowledge:
            for iteration in range(min(flow.max_search_iterations, self.settings.max_search_iterations) + 1):
                fresh_queries = [query for query in queries if query not in search_log]
                if not fresh_queries:
                    break
                for query in fresh_queries:
                    search_log.append(query)
                    for result in self.knowledge.search(
                        query,
                        flow.id,
                        source_name=ORGANIZATIONAL_DOCUMENT if organizational else None,
                    ):
                        if organizational:
                            result.score += 1.0
                        current = candidates.get(result.section.id)
                        if current is None or result.score > current.score:
                            candidates[result.section.id] = result
                ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
                decision = await self._select_evidence(message, ranked, plan.topics)
                if decision is None:
                    break
                if decision.selected_ids:
                    selected_ids = [value for value in decision.selected_ids if value in candidates]
                else:
                    selected_ids = [item.section.id for item in ranked[:4]]
                if decision.ready or not decision.additional_queries or iteration >= flow.max_search_iterations:
                    break
                queries = self._normalize_queries(decision.additional_queries, flow)
        else:
            selected_ids = []

        if plan.needs_knowledge and candidates and not selected_ids:
            selected_ids = [item.section.id for item in sorted(candidates.values(), key=lambda item: item.score, reverse=True)[:4]]
        allowed = self.knowledge.flow_sections.get(flow.id, set())
        evidence = self.knowledge.expand(selected_ids, allowed)
        packed_evidence = self._pack_evidence(evidence)
        answer = await self._answer(message, history, flow, packed_evidence)
        self.memory.append(
            resolved_session,
            ChatMessage(role="user", content=message),
            ChatMessage(role="assistant", content=answer),
        )
        source_scores = {item.section.id: item.score for item in candidates.values()}
        sources = [
            SourceReference(
                id=section.id,
                document=section.source.name,
                title=section.document_title,
                section=" > ".join(section.heading_path),
                lines=f"{section.start_line}-{section.end_line}",
                score=source_scores.get(section.id),
            )
            for section in evidence
        ]
        include_debug = debug and self.settings.debug_responses
        return ChatResponse(
            answer=answer,
            session_id=resolved_session,
            flow_id=flow.id,
            model=self.provider.model,
            sources=sources,
            debug={
                "plan": plan.model_dump(),
                "queries": search_log,
                "candidate_ids": list(candidates),
                "selected_ids": selected_ids,
            } if include_debug else None,
        )

    async def _plan(self, question: str, history: list[ChatMessage], flow: FlowConfig) -> SearchPlan:
        prompt = (
            "Você planeja buscas em uma base corporativa. Retorne SOMENTE JSON válido com: "
            '{"queries":["..."],"topics":["..."],"needs_knowledge":true}. '
            "Crie até 3 consultas curtas e independentes, uma para cada assunto da pergunta, preservando valores e nomes. "
            "Quando houver valor monetário e aprovação, inclua uma consulta geral por alçadas e faixas de aprovação, "
            "pois a faixa aplicável precisa ser recuperada antes de responder. "
            "Quando a pergunta mencionar pessoa, cargo, GM, General Manager, gerente geral, gerente, diretor, líder, responsável, área, departamento, "
            "organograma ou quem é de uma área, inclua uma consulta específica por pessoa/cargo/área e outra por "
            "Management Team Brazil ou RAG_SQMS_Organizacional. Priorize o documento organizacional. "
            "Para saudação ou conversa comum, use needs_knowledge=false. Não responda à pergunta."
        )
        context = "\n".join(f"{item.role}: {item.content}" for item in history[-4:])
        try:
            result = await self.provider.complete_json([
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=f"Fluxo: {flow.name}\nHistórico:\n{context}\nPergunta: {question}"),
            ], SearchPlan,
                thinking=self.settings.llm_thinking_json,
                max_tokens=self.settings.llm_max_tokens_json)
        except Exception:
            result = None
        return result or SearchPlan(
            queries=[question],
            topics=[],
            needs_knowledge=self._looks_like_knowledge(question) or self._is_organizational(question),
        )

    async def _select_evidence(
        self,
        question: str,
        candidates: list[RankedSection],
        topics: list[str],
    ) -> EvidenceDecision | None:
        if not candidates:
            return None
        catalog = "\n\n".join(
            f"ID: {item.section.id}\nSeção: {' > '.join(item.section.heading_path)}\n"
            f"Trecho:\n{item.section.content[:1200]}"
            for item in candidates[:10]
        )
        prompt = (
            "Selecione evidências para responder à pergunta. Retorne SOMENTE JSON válido: "
            '{"selected_ids":["id"],"additional_queries":[],"ready":true}. '
            "Selecione todas as seções complementares necessárias. Se faltar um assunto, "
            "defina ready=false e crie até 2 consultas adicionais. Não responda à pergunta."
        )
        try:
            return await self.provider.complete_json([
                ChatMessage(role="system", content=prompt),
                ChatMessage(role="user", content=f"Pergunta: {question}\nTópicos: {topics}\n\nCandidatos:\n{catalog}"),
            ], EvidenceDecision,
                thinking=self.settings.llm_thinking_json,
                max_tokens=self.settings.llm_max_tokens_json)
        except Exception:
            return None

    async def _answer(
        self,
        question: str,
        history: list[ChatMessage],
        flow: FlowConfig,
        evidence: str,
    ) -> str:
        evidence_instruction = (
            "Use as evidências abaixo como fonte para fatos corporativos. Combine seções quando necessário. "
            "Não invente cargos, valores, regras ou etapas. Não exponha mecanismos de busca, RAG, contexto, "
            "fragmentos ou limitações dos documentos. Não diga 'não encontrei' ou 'não está detalhado'. "
            "Escreva de forma profissional e objetiva, sem usar emojis, ícones decorativos ou emoticons. "
            "Para perguntas organizacionais, informe sempre o nome, cargo e área exatamente como aparecem na evidência. "
            "Se a pergunta for sobre quem responde a quem, só informe a relação se ela estiver explicitamente escrita "
            "na evidência; caso contrário, diga que o documento não detalha a subordinação e não infira hierarquia. "
            "Se faltar um dado essencial, diga o que é possível orientar e faça uma pergunta curta para continuar."
            if evidence else
            "Nenhuma evidência corporativa foi selecionada. Se for saudação ou conversa casual, responda "
            "natural e breve, sem emojis. Se a pergunta for sobre SQMS/compras/aprovações mas só faltar evidência, faça "
            "uma pergunta curta para esclarecer, sem inventar informações. Se for qualquer outro assunto, "
            "siga a regra de escopo do system prompt: recuse o conteúdo e ofereça no que pode ajudar."
        )
        messages = [
            ChatMessage(role="system", content=f"{flow.system_prompt}\n\n{evidence_instruction}"),
            *history[-self.settings.max_history_messages:],
            ChatMessage(
                role="user",
                content=f"Pergunta atual: {question}\n\nEVIDÊNCIAS CORPORATIVAS:\n{evidence or '(nenhuma)'}",
            ),
        ]
        answer = await self.provider.complete(
            messages,
            temperature=0.25,
            thinking=self.settings.llm_thinking_answer,
            max_tokens=self.settings.llm_max_tokens_answer,
        )
        return strip_emojis(answer)

    def _pack_evidence(self, sections: list[DocumentSection]) -> str:
        max_chars = self.settings.max_evidence_tokens * 4
        packed: list[str] = []
        size = 0
        for section in sections:
            text = section.citation_text
            if size + len(text) > max_chars:
                remaining = max_chars - size
                if remaining > 500:
                    packed.append(text[:remaining])
                break
            packed.append(text)
            size += len(text)
        return "\n\n---\n\n".join(packed)

    @staticmethod
    def _looks_like_knowledge(question: str) -> bool:
        normalized = question.lower()
        return bool(re.search(r"\b(sqms|cotacao|cotação|compra|pedido|aprova|notebook|computador|procedimento|capex|fornecedor)\b", normalized))

    @staticmethod
    def _normalize_queries(queries: list[str], flow: FlowConfig) -> list[str]:
        normalized: list[str] = []
        for query in queries:
            value = query.strip()
            for canonical, aliases in flow.aliases.items():
                if any(alias.lower() in value.lower() for alias in aliases) and canonical.lower() not in value.lower():
                    value = f"{value} {canonical}"
            if value and value not in normalized:
                normalized.append(value)
        return normalized[:4]

    @staticmethod
    def _is_organizational(question: str) -> bool:
        normalized = question.lower()
        return bool(re.search(
            r"\b(pessoa|cargo|gerente|diretor|diretora|líder|lider|responsável|responsavel|"
            r"área|area|departamento|organograma|quem é|quem e|quem responde|lidera|gm|manager|general manager|gerente geral|director|leader)\b",
            normalized,
        ))

    @staticmethod
    def _normalize_organizational_query(query: str) -> str:
        normalized = re.sub(r'\bt\s*\.\s*i\.?(?!\w)', 'TI', query, flags=re.IGNORECASE)
        normalized = re.sub(r'\bi\s*\.\s*t\.?(?!\w)', 'IT', normalized, flags=re.IGNORECASE)
        return normalized
