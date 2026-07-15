//! Reciprocal Rank Fusion.
//!
//! RRF combines multiple ranked lists without needing comparable score scales.
//! For each candidate document `d` appearing at 0-based rank `r_i` in list `i`:
//!
//! ```text
//! RRF(d) = Σ_i  1 / (k + r_i)
//! ```
//!
//! with `k` typically 60. Because BM25 scores and cosine similarities live on
//! completely different scales, RRF is the robust default for hybrid search.

use crate::bm25::Bm25Hit;
use crate::vector::VectorHit;
use std::collections::HashMap;

/// Tunable parameters for RRF.
#[derive(Debug, Clone, Copy)]
pub struct RrfConfig {
    /// Smoothing constant (the original paper uses 60).
    pub k: f32,
    /// Optional weight on the BM25 signal (the other weight is `1.0 - bm25_weight`).
    /// Set to `0.5` for equal fusion.
    pub bm25_weight: f32,
    /// How deep each backend should retrieve before fusion (candidate window).
    pub candidate_window: usize,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            bm25_weight: 0.5,
            candidate_window: 50,
        }
    }
}

impl RrfConfig {
    pub fn new(k: f32, bm25_weight: f32) -> Self {
        Self {
            k,
            bm25_weight,
            candidate_window: 50,
        }
    }

    /// Bias toward lexical matches.
    pub fn lexical_heavy() -> Self {
        Self::new(60.0, 0.7)
    }

    /// Bias toward semantic matches.
    pub fn semantic_heavy() -> Self {
        Self::new(60.0, 0.3)
    }

    pub fn with_candidate_window(mut self, w: usize) -> Self {
        self.candidate_window = w.max(1);
        self
    }
}

/// A fused hit: the id, the combined RRF score, and the per-signal ranks
/// (0-based, `None` if the document was absent from that signal's list).
#[derive(Debug, Clone)]
pub struct FusedHit {
    pub id: String,
    pub score: f32,
    pub bm25_rank: Option<usize>,
    pub vector_rank: Option<usize>,
}

/// Fuse BM25 and vector hits with weighted RRF.
///
/// Hits are returned sorted by descending fused score.
pub fn rrf(bm25: &[Bm25Hit], vectors: &[VectorHit], cfg: &RrfConfig) -> Vec<FusedHit> {
    let w_b = cfg.bm25_weight.clamp(0.0, 1.0);
    let w_v = 1.0 - w_b;

    // Map id -> (bm25_rank, vector_rank).
    let mut map: HashMap<String, (Option<usize>, Option<usize>)> = HashMap::new();
    for (rank, h) in bm25.iter().enumerate() {
        map.entry(h.id.clone())
            .or_insert((None, None))
            .0 = Some(rank);
    }
    for (rank, h) in vectors.iter().enumerate() {
        map.entry(h.id.clone())
            .or_insert((None, None))
            .1 = Some(rank);
    }

    let mut out: Vec<FusedHit> = map
        .into_iter()
        .map(|(id, (b_rank, v_rank))| {
            let mut score = 0.0f32;
            if let Some(r) = b_rank {
                score += w_b / (cfg.k + r as f32);
            }
            if let Some(r) = v_rank {
                score += w_v / (cfg.k + r as f32);
            }
            FusedHit {
                id,
                score,
                bm25_rank: b_rank,
                vector_rank: v_rank,
            }
        })
        .collect();

    out.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(id: &str, score: f32) -> VectorHit {
        VectorHit { id: id.to_string(), score }
    }
    fn b(id: &str, score: f32) -> Bm25Hit {
        Bm25Hit { id: id.to_string(), score }
    }

    #[test]
    fn fuses_overlapping_and_unique() {
        let bm25 = vec![b("a", 5.0), b("b", 3.0), b("c", 1.0)];
        let vectors = vec![v("b", 0.9), v("c", 0.8), v("d", 0.7)];
        let fused = rrf(&bm25, &vectors, &RrfConfig::default());
        // 'b' and 'c' appear in both lists, so they outrank single-list members.
        let ids: Vec<_> = fused.iter().map(|h| h.id.as_str()).collect();
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"d"));
        // 'b' has the best combined rank (rank 1 + rank 0).
        assert_eq!(fused[0].id, "b");
    }

    #[test]
    fn empty_lists_yield_empty() {
        let fused = rrf(&[], &[], &RrfConfig::default());
        assert!(fused.is_empty());
    }

    #[test]
    fn weight_changes_winner() {
        // 'a' wins lexically (rank 0), 'b' wins semantically (rank 0).
        let bm25 = vec![b("a", 5.0), b("b", 1.0)];
        let vectors = vec![v("b", 0.9), v("a", 0.1)];

        let lex = rrf(&bm25, &vectors, &RrfConfig::lexical_heavy());
        assert_eq!(lex[0].id, "a");

        let sem = rrf(&bm25, &vectors, &RrfConfig::semantic_heavy());
        assert_eq!(sem[0].id, "b");
    }
}