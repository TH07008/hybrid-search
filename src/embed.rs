//! Lightweight, **deterministic** embedding function.
//!
//! Real RAG pipelines pass text through an embedding model (e.g. MiniLM, E5)
//! and feed the resulting dense vector into [`crate::VectorIndex`]. To keep the
//! CLI demo self-contained and reproducible without downloading a model, this
//! module provides a fast bag-of-words / hashing embedding:
//!
//! * Each token is hashed into `dim` buckets with a signed component.
//! * The document vector is the sum of token vectors, which approximates a
//!   semantic "topic fingerprint": overlapping vocabulary raises cosine
//!   similarity, which is exactly what the vector side of hybrid search needs
//!   to demonstrate the fusion behavior.
//!
//! This is **not** a production embedding — it has no notion of synonyms or
//! paraphrase. Use real model embeddings for real retrieval quality; this
//! function only exists so `hybrid-search demo` runs end-to-end offline.

use fxhash::FxHasher;
use std::hash::{Hash, Hasher};

/// Tokenize like the BM25 side: lowercase, split on non-alphanumeric.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Produce a `dim`-dimensional hashing embedding for `text`.
pub fn hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    for token in tokenize(text) {
        // Two independent hashes: one for the bucket index, one for the sign.
        let mut h_pos = FxHasher::default();
        token.hash(&mut h_pos);
        h_pos.write_u8(0);
        let bucket = (h_pos.finish() as usize) % dim;

        let mut h_sign = FxHasher::default();
        token.hash(&mut h_sign);
        h_sign.write_u8(1);
        let sign = if (h_sign.finish() & 1) == 0 { -1.0f32 } else { 1.0f32 };

        vec[bucket] += sign;
    }
    vec
}