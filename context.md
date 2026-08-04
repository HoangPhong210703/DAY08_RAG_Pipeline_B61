# Context — Task Progress Tracker

Tracks status of the Day 8 RAG Pipeline lab (`LAB_GUIDE.md` / `README.md`) against the
actual state of this repo. Update this file whenever a task's status changes.

**Workflow: read the relevant `TestTaskN` class in `tests/test_individual.py` *before*
implementing a task**, not after — it pins down the exact function names, arg names/
defaults (e.g. `top_k`, `score_threshold`), and required return-dict keys (`content`,
`score`, `metadata`, `source`, ...) each task must expose. Coding against the test
upfront avoids a second pass to rename/reshape things afterward.

**Then, after implementing, run its tests before marking it ✅** —
`pytest tests/test_individual.py::TestTaskN -v`. A task is only "done" once its tests
pass, not just when the code is written (the suite `skipTest`s on `NotImplementedError`/
`ImportError` instead of failing, so a stub can look inert but still be incomplete).

Legend: ✅ done & tested · 🚧 in progress / partially done · ⬜ not started

## Pipeline Tasks (Task 1–10, 50 pts)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Collect legal docs | ✅ | `src/task1_collect_legal_docs.py` pulls 5 real documents from HF dataset [`tmquan/vbpl-vn`](https://huggingface.co/datasets/tmquan/vbpl-vn) (via Datasets Server REST API) and renders them as real PDFs (fpdf2 + Arial Unicode font) into `data/landing/legal/`. **Not wired into the retrieval pipeline** — Task 4 reads straight from the sqlite corpus instead. `TestTask1`: 3/3 pass. |
| 2 | Crawl news articles | ⬜ | `src/task2_crawl_news.py` still a stub (`ARTICLE_URLS` empty, `crawl_article` raises `NotImplementedError`). `data/landing/news/` empty. |
| 3 | Convert to Markdown | ⬜ | `src/task3_convert_markdown.py` still a stub (`convert_legal_docs`/`convert_news_articles` raise `NotImplementedError`). `data/standardized/news/` empty; nothing consumes the Task 1 PDFs yet. |
| 4 | Chunking & Indexing | ✅ | `src/task4_chunking_indexing.py` reads the **`chunks` table** of `data/standardized/legal/ragvbpl.sqlite` (pre-built by the separate `data_ingestion/` pipeline, structure-aware Điều/Khoản/Điểm chunks), **filtered to `legal_domains` containing `"labor"`** (881 rows / 7607 total → 1800 chunks after 800-char/100-overlap re-split). Embeds via **OpenAI `text-embedding-3-small`** (1536-dim, not local `bge-m3` — CPU-only machine, no GPU) and indexes into ChromaDB (`chroma_db/`, collection `university_services_docs`). `TestTask4`: 4/4 pass. Index has been built and verified with real queries. |
| 5 | Semantic Search | ✅ | `src/task5_semantic_search.py::semantic_search()` implemented, queries the Task 4 collection using the same embedding model. `TestTask5`: 4/4 pass. |
| 6 | Lexical Search (BM25) | ⬜ | `src/task6_lexical_search.py` still a stub. |
| 7 | Reranking (RRF) | ⬜ | `src/task7_reranking.py` still a stub. |
| 8 | PageIndex Vectorless | ⬜ | `src/task8_pageindex_vectorless.py` still a stub. |
| 9 | Retrieval Pipeline | ⬜ | `src/task9_retrieval_pipeline.py::retrieve()` still a stub — depends on Task 6–8. |
| 10 | Generation + Citation | ⬜ | `src/task10_generation.py` still a stub — `app.py` already calls `generate_with_citation()` but it isn't implemented yet. |

## Group Project (30 pts)

| Item | Status | Notes |
|------|--------|-------|
| Streamlit Chatbot (`app.py`) | 🚧 | UI shell exists (sidebar, chat history, suggestions), calls `task10_generation.generate_with_citation` — blocked on Task 9/10. |
| `group_project/evaluation/golden_dataset.json` | 🚧 | Only 3 Q&A pairs; lab requires ≥15. |
| `group_project/evaluation/eval_pipeline.py` | 🚧 | File exists (RAGAS-based), not run yet — blocked on Task 9/10. |
| `group_project/evaluation/results.md` | ⬜ | Template only, no results filled in. |

## Environment / Setup (CP0)

| Item | Status | Notes |
|------|--------|-------|
| `.venv/` | ✅ | Created; installed: `chromadb`, `openai`, `fpdf2`, `langchain-text-splitters`, `pypdf`, `pytest`, `python-dotenv`. (`sentence-transformers`/`torch` were installed then **removed** once we switched to OpenAI embeddings.) |
| `.env` | ✅ | Exists with a working `OPENAI_API_KEY`. |

## Key deviations from the lab spec (intentional, by user direction)

- **Legal corpus**: not RMIT tuition/scholarship/dorm PDFs as the lab suggests — uses the
  pre-existing `data_ingestion/` pipeline's output, a general Vietnamese National Legal
  Database corpus (`ragvbpl.sqlite`, sourced from HF datasets `tmquan/vbpl-vn`,
  `th1nhng0/vietnamese-legal-documents`, `undertheseanlp/UTS_VLC`,
  `tmquan/phapdien-moj-gov-vn`), narrowed to the **"labor" (Lao Động) domain only**.
- **Embedding model**: OpenAI `text-embedding-3-small` via API, not local `BAAI/bge-m3` —
  avoids a large local model download / CPU-only embedding run.
- **Task 1 PDFs are cosmetic**: they satisfy the Task 1 test's file-existence check but are
  a separate, disconnected sample from the same underlying dataset family — not consumed
  by Task 3/4/5.

## Suggested next steps

1. Task 6 (BM25 lexical search) — straightforward, no external deps beyond `rank-bm25`.
2. Task 7 (RRF reranking) — pure logic, no new deps.
3. Task 9 (retrieve pipeline) — wires 5+6+7 together; needs the cosine-vs-RRF threshold
   trap from `LAB_GUIDE.md` handled correctly.
4. Task 10 (generation + citation) — needs an LLM API key (OpenRouter per `.env.example`,
   or reuse `OPENAI_API_KEY`).
5. Task 8 (PageIndex) — optional/fallback-only; lowest priority, needs its own API key.
6. Task 2/3 — only needed for the news domain and to fully satisfy CP1; not blocking the
   legal-only retrieval pipeline.
