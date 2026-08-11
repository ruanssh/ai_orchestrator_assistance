# SQMS AI Orchestrator

Serviço FastAPI modular para chat com LLMs OpenAI-compatible, recuperação de conhecimento e fluxos corporativos. A Lity é o provider inicial, mas não está acoplada ao núcleo.

## Como funciona

1. O Markdown é dividido por títulos, preservando hierarquia, listas, metadados e linhas.
2. A LLM cria consultas de busca em JSON.
3. BM25 recupera seções por termos exatos.
4. Embeddings e reranking podem complementar a busca.
5. A LLM seleciona evidências e pode solicitar outra busca.
6. Seções relacionadas são expandidas dentro do orçamento de contexto.
7. Uma chamada final responde naturalmente e retorna as fontes.

O projeto não depende de `tool calling` nativo.

## Execução

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn sqms_ai_orchestrator.main:app --reload --port 8200
```

Documentação interativa: `http://localhost:8200/docs`.

Com Docker:

```bash
docker compose up -d --build
```

O Compose monta `../procedures_rag` em `/procedures_rag`, caminho esperado pelo fluxo `procurement` dentro do container.

## Teste da busca antes da Lity

```bash
curl -s http://localhost:8200/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"como criar uma cotação no SQMS","flow_id":"procurement"}'
```

Esse endpoint permite validar recuperação sem misturar o resultado com geração da LLM.

## Chat

```bash
curl -s http://localhost:8200/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Quero cotar um notebook de R$ 6.000. Como faço e quem aprova?","flow_id":"procurement"}'
```

## Recuperação semântica

A busca lexical funciona por padrão e não exige GPU. Para habilitar BGE-M3 e reranking:

```bash
pip install -e ".[semantic]"
```

```env
SQMS_AI_SEMANTIC_ENABLED=true
SQMS_AI_EMBEDDING_MODEL=BAAI/bge-m3
SQMS_AI_RERANK_ENABLED=true
SQMS_AI_RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

Os modelos são baixados na primeira inicialização. Em CPU, comece somente com embeddings; o reranker pode aumentar bastante a latência.

## Novo caso de uso

Crie `flows/<id>/config.yaml`:

```yaml
id: quality
name: Qualidade
knowledge_paths:
  - ../procedures_quality
max_search_iterations: 2
aliases: {}
system_prompt: |
  Você é um assistente corporativo especializado em Qualidade.
```

Não é necessário alterar o orquestrador, o provider ou os retrievers.

## Variáveis principais

- `SQMS_AI_LLM_BASE_URL`: raiz OpenAI-compatible, sem `/chat/completions`.
- `SQMS_AI_LLM_API_KEY`: chave enviada como Bearer.
- `SQMS_AI_LLM_MODEL`: modelo do endpoint.
- `SQMS_AI_LLM_CONTEXT_TOKENS`: janela declarada do modelo.
- `SQMS_AI_MAX_EVIDENCE_TOKENS`: orçamento reservado às evidências.
- `SQMS_AI_DEBUG_RESPONSES`: permite diagnóstico quando o request envia `debug: true`.

O diagnóstico fica desativado por padrão para não expor decisões internas aos usuários.
