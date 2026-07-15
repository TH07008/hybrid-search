//! # hybrid-search
//!
//! A fast, in-memory **hybrid search index** for RAG developers.
//!
//! It combines two complementary retrieval signals:
//!
//! * **BM25 keyword search** via [`tantivy`] — sparse lexical matching, great for
//!   exact terms, identifiers, rare words, and queries that don't share semantics
//!   with the target document.
//! * **Vector semantic search** via [`nalgebra`] — dense cosine similarity, great
//!   for paraphrasing and concept-level matching.
//!
//! Both ranked lists are merged with **Reciprocal Rank Fusion (RRF)**, a robust
//! rank-based combination that needs no score calibration between the two systems.
//!
//! ```
//! use hybrid_search::{HybridIndex, SearchRequest};
//!
//! let mut idx = HybridIndex::in_memory(4);
//! idx.add_document("d1", "The rust programming language is fast.", vec![0.9, 0.1, 0.0, 0.0]);
//! idx.add_document("d2", "Cars can rust when left in the rain.",  vec![0.1, 0.9, 0.0, 0.0]);
//! idx.commit().unwrap();
//!
//! let res = idx.search(&SearchRequest::new("rust language", vec![0.9, 0.1, 0.0, 0.0]).top_k(2)).unwrap();
//! assert_eq!(res[0].id, "d1");
//! ```

pub mod bm25;
pub mod vector;
pub mod fusion;
pub mod embed;
mod err;

pub use err::{HybridError, Result};
pub use bm25::{Bm25Index, Bm25Hit};
pub use vector::{VectorIndex, VectorHit};
pub use fusion::{rrf, RrfConfig, FusedHit};
pub use embed::hash_embed;

use serde::{Deserialize, Serialize};

/// A document to be indexed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Stable identifier. Must be unique across the index.
    pub id: String,
    /// The free-form text body used for BM25 tokenization.
    pub text: String,
    /// Dense embedding vector used for semantic search.
    /// All documents in one index must share the same dimensionality.
    pub vector: Vec<f32>,
}

impl Document {
    pub fn new(id: impl Into<String>, text: impl Into<String>, vector: Vec<f32>) -> Self {
        Self { id: id.into(), text: text.into(), vector }
    }
}

/// A search request.
///
/// Both `query_text` and `query_vector` may be supplied; the hybrid index will
/// run BM25 on the text and cosine search on the vector, then fuse the rankings.
/// If a signal is omitted, only the provided signal is used.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query_text: Option<String>,
    pub query_vector: Option<Vec<f32>>,
    pub top_k: usize,
    pub rrf: RrfConfig,
    /// If true, the fused score field carries the RRF score; hits are always
    /// sorted descending by fused score.
    pub include_scores: bool,
}

impl SearchRequest {
    pub fn text(query: impl Into<String>) -> Self {
        Self {
            query_text: Some(query.into()),
            query_vector: None,
            top_k: 10,
            rrf: RrfConfig::default(),
            include_scores: true,
        }
    }

    pub fn vector(query: Vec<f32>) -> Self {
        Self {
            query_text: None,
            query_vector: Some(query),
            top_k: 10,
            rrf: RrfConfig::default(),
            include_scores: true,
        }
    }

    pub fn new(query_text: impl Into<String>, query_vector: Vec<f32>) -> Self {
        Self {
            query_text: Some(query_text.into()),
            query_vector: Some(query_vector),
            top_k: 10,
            rrf: RrfConfig::default(),
            include_scores: true,
        }
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k.max(1);
        self
    }

    pub fn with_rrf(mut self, cfg: RrfConfig) -> Self {
        self.rrf = cfg;
        self
    }
}

/// A fused search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    /// Reciprocal Rank Fusion score (higher = better).
    pub score: f32,
    /// BM25 rank (0-based, `None` if not returned by the lexical index).
    pub bm25_rank: Option<usize>,
    /// Vector rank (0-based, `None` if not returned by the semantic index).
    pub vector_rank: Option<usize>,
    /// Raw BM25 score (if available).
    pub bm25_score: Option<f32>,
    /// Raw cosine similarity (if available).
    pub vector_score: Option<f32>,
}

/// The top-level hybrid index.
///
/// Wraps a [`tantivy`] in-memory index for BM25 and a [`VectorIndex`] for
/// cosine similarity, and exposes a single fused [`HybridIndex::search`] API.
pub struct HybridIndex {
    bm25: Bm25Index,
    vectors: VectorIndex,
    dim: usize,
    /// Documents kept in insertion order for cheap text lookup by id.
    docs: std::collections::HashMap<String, String>,
}

impl HybridIndex {
    /// Create an empty in-memory index expecting `dim`-dimensional vectors.
    pub fn in_memory(dim: usize) -> Self {
        Self {
            bm25: Bm25Index::in_memory(),
            vectors: VectorIndex::new(dim),
            dim,
            docs: std::collections::HashMap::new(),
        }
    }

    /// Vector dimensionality this index expects.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of documents currently indexed.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Add a document. The vector length must match [`Self::dim`].
    ///
    /// The document is buffered until [`Self::commit`] is called.
    pub fn add_document(&mut self, id: impl Into<String>, text: impl Into<String>, vector: Vec<f32>) -> Result<()> {
        let id = id.into();
        let text = text.into();
        if vector.len() != self.dim {
            return Err(HybridError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.docs.contains_key(&id) {
            return Err(HybridError::DuplicateId(id));
        }
        self.bm25.add_document(&id, &text)?;
        self.vectors.add(&id, vector)?;
        self.docs.insert(id, text);
        Ok(())
    }

    /// Add a full [`Document`].
    pub fn add(&mut self, doc: Document) -> Result<()> {
        self.add_document(doc.id, doc.text, doc.vector)
    }

    /// Finalize the BM25 segment so it is searchable.
    pub fn commit(&mut self) -> Result<()> {
        self.bm25.commit()
    }

    /// Run a hybrid search and return the fused top-k hits.
    pub fn search(&self, req: &SearchRequest) -> Result<Vec<SearchResult>> {
        let mut bm25_hits: Vec<Bm25Hit> = Vec::new();
        let mut vector_hits: Vec<VectorHit> = Vec::new();

        if let Some(q) = &req.query_text {
            // BM25 retrieves a wider candidate set so fusion has signal overlap.
            let candidate_k = req.top_k.max(req.rrf.candidate_window);
            bm25_hits = self.bm25.search(q, candidate_k)?;
        }
        if let Some(v) = &req.query_vector {
            if v.len() != self.dim {
                return Err(HybridError::DimensionMismatch {
                    expected: self.dim,
                    got: v.len(),
                });
            }
            let candidate_k = req.top_k.max(req.rrf.candidate_window);
            vector_hits = self.vectors.search(v, candidate_k)?;
        }

        // Build fused ranking.
        let fused = rrf(&bm25_hits, &vector_hits, &req.rrf);

        // Collect ids present in either list (union) — already encoded in `fused`.
        let mut results = Vec::with_capacity(fused.len());
        for FusedHit { id, score, bm25_rank, vector_rank } in &fused {
            let bm25_score = bm25_hits
                .iter()
                .find(|h| h.id == *id)
                .map(|h| h.score);
            let vector_score = vector_hits
                .iter()
                .find(|h| h.id == *id)
                .map(|h| h.score);
            let text = self.docs.get(id).cloned().unwrap_or_default();
            results.push(SearchResult {
                id: id.clone(),
                text,
                score: *score,
                bm25_rank: *bm25_rank,
                vector_rank: *vector_rank,
                bm25_score,
                vector_score,
            });
        }

        // Truncate to top_k (rrf already returns sorted, but be defensive).
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(req.top_k);

        // Optionally strip score detail when include_scores is false.
        if !req.include_scores {
            for r in &mut results {
                r.score = 0.0;
                r.bm25_score = None;
                r.vector_score = None;
            }
        }
        Ok(results)
    }
}