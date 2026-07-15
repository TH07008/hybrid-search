//! Vector (semantic) search using cosine similarity over dense vectors,
//! backed by [`nalgebra`].

use crate::Result;
use nalgebra::DVector;

/// A single vector hit.
#[derive(Debug, Clone)]
pub struct VectorHit {
    pub id: String,
    /// Cosine similarity in [-1, 1].
    pub score: f32,
}

/// In-memory vector index.
///
/// Vectors are stored as `f32` rows and queried with cosine similarity.
/// For the typical RAG corpus size (thousands to low millions of chunks)
/// this brute-force scan in Rust is already very fast; the API is designed
/// so a future ANN backend (e.g. usearch / hnsw) can be slotted in behind it.
pub struct VectorIndex {
    dim: usize,
    ids: Vec<String>,
    /// L2-normalized rows; queries are also normalized, so cosine = dot product.
    rows: Vec<DVector<f32>>,
}

impl VectorIndex {
    /// Create an empty index expecting `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "vector dimension must be positive");
        Self {
            dim,
            ids: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// Expected vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Add a vector under `id`. The length must equal [`Self::dim`].
    pub fn add(&mut self, id: &str, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(crate::HybridError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let mut v = DVector::from_vec(vector);
        normalize_in_place(&mut v);
        self.ids.push(id.to_string());
        self.rows.push(v);
        Ok(())
    }

    /// Return the top-`limit` most similar vectors to `query`.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<VectorHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dim {
            return Err(crate::HybridError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut q = DVector::from_vec(query.to_vec());
        normalize_in_place(&mut q);

        // Cosine = dot product because both sides are unit-norm.
        let mut scored: Vec<(f32, usize)> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| (q.dot(row), i))
            .collect();

        // Partial sort: we only need the top `limit`.
        let take = limit.min(scored.len());
        scored.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored[..take].sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored
            .iter()
            .take(limit)
            .map(|(score, i)| VectorHit {
                id: self.ids[*i].clone(),
                score: *score,
            })
            .collect())
    }
}

/// L2-normalize `v` in place. Zero vectors stay zero (cosine defined as 0).
fn normalize_in_place(v: &mut DVector<f32>) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.scale_mut(1.0 / n);
    }
}