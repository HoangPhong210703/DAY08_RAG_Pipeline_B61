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
| 6 | Lexical Search (BM25) | ✅ | `src/task6_lexical_search.py::lexical_search()` implemented (`rank-bm25`, lazy-cached index built over the **same Task 4 corpus** — `load_documents()` + `chunk_documents()`, 1800 labor-law chunks). `TestTask6`: 1 passed, 3 skipped — the 3 skips are **expected**, not bugs: those tests query in English ("tuition fee", "scholarship eligibility", "library study room") against a pure-Vietnamese corpus, so BM25 (literal token overlap) legitimately finds 0 matches and the test itself calls `skipTest` on empty results. Verified working with a real Vietnamese query (`"bảo hiểm xã hội thai sản"` → correct maternity-benefit articles, top score 12.5). |
| 7 | Reranking (RRF) | ✅ | `src/task7_reranking.py` implemented: `rerank_rrf(ranked_lists, top_k, k=60)` fuses N ranked lists by `Σ 1/(k+rank)` (dedup by `content`). Top-level `rerank(query, candidates, top_k, method="rrf")` — the interface the grader calls directly with **one flat list** — sorts that list by its own `score` first, then runs it through `rerank_rrf([ranked])` as the degenerate 1-list case; **use `rerank_rrf([dense, sparse])` directly for the real multi-ranker merge** (that's what Task 9 will do before calling `rerank()` for the final pass). `cross_encoder`/`mmr` methods left as stubs (not required by tests, optional/bonus). `TestTask7`: 3/3 pass. Manually verified RRF fusion ranks items appearing in both lists above single-list-only items. |
| 8 | PageIndex Vectorless | ✅ | **Not the real PageIndex SDK** — no `PAGEINDEX_API_KEY` available (would need a pageindex.ai account, a decision left to the user; they chose the local alternative). Instead `src/task8_pageindex_vectorless.py::pageindex_search()` reimplements the same *principle* PageIndex advertises — navigate document structure (document → chapter → article title path) and match the query against **titles only**, no chunking/embedding — using the `parsed_articles` JOIN `documents` tables in `ragvbpl.sqlite` (720 unchunked, full-text article nodes, `labor` domain), returning full `article_text` for the best title matches. Verified with real Vietnamese queries (all 3 top results were the exactly-correct articles). `TestTask8`: 2/2 pass. Task 9's fallback now calls this for real (previously always degraded to hybrid via the `except`); for genuinely nonsensical queries it correctly still returns `[]` (no title-token overlap) and Task 9 falls through to hybrid results — verified, not a bug. |
| 9 | Retrieval Pipeline | ✅ | `src/task9_retrieval_pipeline.py::retrieve()` implemented: semantic (Task 5) + lexical (Task 6) → `rerank_rrf` merge → `rerank()` final pass → fallback check against the **original cosine score** (not RRF score — avoids the documented trap). `pageindex_search` (Task 8, still unimplemented) is called in a `try/except`, so fallback gracefully degrades back to hybrid results instead of crashing when it's unavailable. `SCORE_THRESHOLD` empirically calibrated to **0.25** by measuring top-1 cosine scores for 5 relevant vs. 5 off-topic Vietnamese queries on this actual corpus/embedding — found relevant (0.35–0.49) and topically-off-topic-but-fluent (0.33–0.42) scores **overlap almost entirely**; the only clean gap is coherent text vs. genuine gibberish (0.17). Documented as a real limitation of cosine-threshold fallback with this embedding, not a bug — threshold is set just above the gibberish band so fallback only fires for truly nonsensical input. `TestTask9`: 4/4 pass. |
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

~~1. Task 6 (BM25 lexical search)~~ ✅ done.
~~2. Task 7 (RRF reranking)~~ ✅ done.
~~3. Task 9 (retrieve pipeline)~~ ✅ done.
~~4. Task 8 (PageIndex)~~ ✅ done (local structure-aware fallback, see table above —
   no PageIndex API key available).
5. **Task 10 (generation + citation)** — next up. Needs an LLM API key (OpenRouter per
   `.env.example`, or reuse `OPENAI_API_KEY`).
6. Task 2/3 — only needed for the news domain and to fully satisfy CP1; not blocking the
   legal-only retrieval pipeline.
7. Group project (chatbot wiring, golden dataset, RAGAS eval) — blocked on Task 10.
