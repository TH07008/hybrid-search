//! BM25 keyword search backed by a [`tantivy`] in-memory index.

use crate::Result;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::schema::document::OwnedValue;
use tantivy::{doc, IndexBuilder, IndexWriter, ReloadPolicy, TantivyDocument};

/// A single BM25 hit.
#[derive(Debug, Clone)]
pub struct Bm25Hit {
    pub id: String,
    pub score: f32,
}

/// In-memory BM25 index.
pub struct Bm25Index {
    index: tantivy::Index,
    writer: Option<IndexWriter>,
    id_field: Field,
    body_field: Field,
}

impl Bm25Index {
    /// Create a fresh in-memory index (RAM directory, nothing persisted).
    pub fn in_memory() -> Self {
        let mut schema = Schema::builder();
        let id_field = schema.add_text_field("id", TEXT | STORED);
        let body_field = schema.add_text_field("body", TEXT);
        let schema = schema.build();

        let index = IndexBuilder::new()
            .schema(schema)
            .create_in_ram()
            .expect("create in-memory tantivy index");

        // The writer requires enough RAM; 15 MiB is fine for demos and small corpora.
        let writer = index.writer(15_000_000).expect("open tantivy writer");

        Self {
            index,
            writer: Some(writer),
            id_field,
            body_field,
        }
    }

    /// Add a document (buffered until [`Self::commit`]).
    pub fn add_document(&mut self, id: &str, text: &str) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| crate::HybridError::Tantivy("writer already finalized".into()))?;
        writer.add_document(doc!(
            self.id_field => id,
            self.body_field => text
        ))?;
        Ok(())
    }

    /// Finalize the current segment so documents become searchable.
    pub fn commit(&mut self) -> Result<()> {
        if let Some(mut w) = self.writer.take() {
            w.commit()?;
            // Drop explicitly so the index can be read consistently.
            drop(w);
        }
        Ok(())
    }

    /// Run a BM25 query and return up to `limit` hits sorted by score.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Bm25Hit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let searcher = reader.searcher();

        let parser = QueryParser::for_index(&self.index, vec![self.body_field]);
        // Lenient parsing: a stray `-` or unknown token won't abort the query.
        let query = parser.parse_query_lenient(query).0;

        let top = TopDocs::with_limit(limit);
        let found = searcher.search(&query, &top)?;

        let mut hits = Vec::with_capacity(found.len());
        for (score, doc_addr) in found {
            let doc: TantivyDocument = searcher.doc(doc_addr)?;
            if let Some(id_val) = doc.get_first(self.id_field) {
                if let OwnedValue::Str(id) = id_val {
                    hits.push(Bm25Hit {
                        id: id.clone(),
                        score,
                    });
                }
            }
        }
        Ok(hits)
    }
}

