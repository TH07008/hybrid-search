# hybrid-search

A fast, in-memory **hybrid search index** for RAG developers, written in Rust.

It combines two complementary retrieval signals and fuses them with **Reciprocal
Rank Fusion (RRF)**:

| Signal | Engine | Strength |
|-------|--------|----------|
| Keyword (lexical) | [`tantivy`](https://crates.io/crates/tantivy) — BM25 | Exact terms, identifiers, rare words, queries with no semantic overlap |
| Semantic (dense) | [`nalgebra`](https://crates.io/crates/nalgebra) — cosine similarity | Paraphrasing, concept-level matching, synonyms |

> **Why hybrid?** Pure vector RAG misses exact keyword matches ("error code
> `E0308`"); pure keyword RAG misses paraphrase. Combining both — at rank
> level, no score calibration — is consistently the most robust retrieval
> strategy. Doing it in Rust makes it orders of magnitude faster than the
> typical Python implementation.

## Features

- 🚀 **In-memory, single process** — no external server, no network hop.
- 🔎 **BM25 via tantivy** — full-text tokenizer, query parser, scoring.
- 🧮 **Cosine vector search via nalgebra** — L2-normalized dot product.
- 🧩 **Reciprocal Rank Fusion** — weighted, tunable `k`, robust to score scale
  differences between BM25 and cosine.
- 📦 **Library + CLI** — embed in your Rust RAG pipeline *or* use as a CLI tool.
- 🧪 **Self-contained demo** — runs end-to-end offline with a deterministic
  hashing embedding (no model download required).

## Architecture

```
            ┌─────────────┐        ┌─────────────┐
   query ──▶│  BM25 index │   ┌──▶│ Vector index│
   text     │  (tantivy)  │   │   │ (nalgebra)  │
            └──────┬──────┘   │   └──────┬──────┘
                   │ ranked   │          │ ranked
                   │ list A   │          │ list B
                   └─────┬────┴──────────┘
                         ▼
                ┌─────────────────┐
                │  Reciprocal Rank │
                │     Fusion       │
                └────────┬────────┘
                         ▼
                  fused top-k hits
```

Both backends retrieve a **candidate window** (default 50) larger than the
requested `top_k`, so the fusion step has enough overlap to combine signals
meaningfully.

## Install / Build

```bash
cargo build --release
# binary at target/release/hybrid-search
```

## CLI

```bash
# Built-in demo on a tiny synthetic corpus (no model needed)
hybrid-search demo

# Index a JSONL corpus (one {"id","text","vector"?} per line).
# If "vector" is omitted, a deterministic hashing embedding is used.
hybrid-search --dim 256 index data/corpus.jsonl --out index.out

# Hybrid search: keyword + auto-embedded query vector
hybrid-search --dim 256 search data/corpus.jsonl "cargo rust build tool" \
    --auto-embed --top-k 3

# Keyword-only search
hybrid-search --dim 256 search data/corpus.jsonl "borrowing" --top-k 3

# Real embeddings: supply the query vector from a JSON file ([0.1, 0.2, ...])
hybrid-search --dim 384 search data/corpus.jsonl "memory safety" \
    --vector query_vec.json --top-k 5 --json

# Tune RRF: k constant + BM25 weight (vector weight = 1 - bm25_weight)
hybrid-search --dim 256 search data/corpus.jsonl "rust language" \
    --auto-embed --rrf-k 30 --bm25-weight 0.7
```

### Sample output

```
query: "cargo rust build tool"  (0.21 ms)
  [0.0167] d7  bm25_rank=Some(0) vec_rank=Some(0)  Cargo is the package manager...
  [0.0161] d9  bm25_rank=Some(3) vec_rank=Some(1)  Asynchronous programming...
  [0.0161] d5  bm25_rank=Some(2) vec_rank=Some(2)  Ownership and borrowing...
```

Each result carries the per-signal ranks so you can see *why* a document ranked
where it did — invaluable for debugging RAG retrieval.

## Library

```rust
use hybrid_search::{HybridIndex, SearchRequest, RrfConfig};

let mut idx = HybridIndex::in_memory(384);
idx.add_document("d1", "The rust programming language is fast.", embed("..."))?;
idx.commit()?;

let req = SearchRequest::new("rust language", query_embed)
    .top_k(5)
    .with_rrf(RrfConfig::new(60.0, 0.5).with_candidate_window(100));

for hit in idx.search(&req)? {
    println!("{:.4}  {}  bm25={:?} vec={:?}",
             hit.score, hit.id, hit.bm25_rank, hit.vector_rank);
}
```

### `SearchRequest` options

| Field | Default | Meaning |
|-------|---------|---------|
| `query_text` | — | Text for BM25; `None` disables the lexical signal |
| `query_vector` | — | Dense vector for cosine search; `None` disables semantic |
| `top_k` | 10 | Final fused result count |
| `rrf.k` | 60.0 | RRF smoothing constant |
| `rrf.bm25_weight` | 0.5 | Weight on BM25 signal (vector weight = `1 − this`) |
| `rrf.candidate_window` | 50 | Per-backend retrieval depth before fusion |

### Tuning the fusion

- `RrfConfig::default()` — balanced.
- `RrfConfig::lexical_heavy()` (`bm25_weight = 0.7`) — favor exact-match
  queries, code identifiers, error codes.
- `RrfConfig::semantic_heavy()` (`bm25_weight = 0.3`) — favor paraphrase /
  concept queries.

## Reciprocal Rank Fusion

For each candidate document `d` appearing at 0-based rank `r_i` in list `i`:

```
RRF(d) = Σ_i  w_i / (k + r_i)
```

RRF operates on **ranks**, not raw scores, so it sidesteps the classic problem
that BM25 scores and cosine similarities live on incomparable scales. The
weighted variant lets you bias toward either signal without calibration.

## How the demo embedding works

`embed::hash_embed` maps each token into a fixed-dimensional signed bag via
hashing. This is **not** a real embedding model — it has no notion of synonyms
or paraphrase. It exists only so `hybrid-search demo` runs offline. For
production retrieval quality, feed real model embeddings (MiniLM, E5, GTE, …)
as the `vector` field.

## Project layout

```
src/
  lib.rs     public API: HybridIndex, SearchRequest, SearchResult, Document
  bm25.rs    tantivy-backed BM25 index
  vector.rs  nalgebra-backed cosine vector index
  fusion.rs  weighted Reciprocal Rank Fusion
  embed.rs   deterministic hashing embedding (demo only)
  err.rs     error types
  main.rs    clap CLI: demo / index / search
data/
  corpus.jsonl   sample corpus
```

## Performance notes

- Vector search is brute-force dot product over L2-normalized rows — already
  very fast in Rust for corpora up to the low millions. For larger scale, the
  `VectorIndex` API is designed to slot in an ANN backend (usearch / hnsw_rs)
  behind it.
- BM25 is fully tantivy (segmented, FST-based term dictionaries) — scales to
  very large corpora with low memory.

## License

MIT