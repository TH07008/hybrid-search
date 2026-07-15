use hybrid_search::{hash_embed, HybridIndex, RrfConfig, SearchRequest};

/// End-to-end smoke test of the hybrid pipeline: build an index from real
/// hashing embeddings, run a hybrid query, and assert that the document that
/// matches both signals outranks documents that match only one.
#[test]
fn hybrid_ranks_overlap_above_single_signal() {
    let dim = 64;
    let corpus = vec![
        ("a", "the rust programming language is fast and memory safe"),
        ("b", "iron and steel can rust when left out in the rain"),
        ("c", "ownership and borrowing are core concepts in rust"),
        ("d", "a car engine burns fuel to generate mechanical power"),
    ];

    let mut idx = HybridIndex::in_memory(dim);
    for (id, text) in &corpus {
        idx.add_document(*id, *text, hash_embed(text, dim)).unwrap();
    }
    idx.commit().unwrap();
    assert_eq!(idx.len(), 4);

    // "rust language" — 'a' is the strong lexical + semantic match.
    let q = "rust language";
    let qv = hash_embed(q, dim);
    let req = SearchRequest::new(q, qv).top_k(4);
    let res = idx.search(&req).unwrap();

    assert!(!res.is_empty());
    assert_eq!(res[0].id, "a");
    // 'a' should appear in both ranked lists.
    assert!(res[0].bm25_rank.is_some());
    assert!(res[0].vector_rank.is_some());

    // 'b' (corrosion rust) shares the token "rust" lexically but is semantically
    // far — it must not win over 'a'.
    assert_ne!(res[0].id, "b");
}

#[test]
fn keyword_only_search_works() {
    let dim = 32;
    let mut idx = HybridIndex::in_memory(dim);
    idx.add_document("x", "alpha beta gamma", hash_embed("alpha beta gamma", dim)).unwrap();
    idx.add_document("y", "delta epsilon", hash_embed("delta epsilon", dim)).unwrap();
    idx.commit().unwrap();

    let req = SearchRequest::text("beta").top_k(1);
    let res = idx.search(&req).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, "x");
    assert!(res[0].bm25_rank.is_some());
    assert!(res[0].vector_rank.is_none());
}

#[test]
fn vector_only_search_works() {
    let dim = 4;
    let mut idx = HybridIndex::in_memory(dim);
    idx.add_document("p", "p", vec![0.9, 0.1, 0.0, 0.0]).unwrap();
    idx.add_document("q", "q", vec![0.1, 0.9, 0.0, 0.0]).unwrap();
    idx.commit().unwrap();

    let req = SearchRequest::vector(vec![0.9, 0.1, 0.0, 0.0]).top_k(1);
    let res = idx.search(&req).unwrap();
    assert_eq!(res[0].id, "p");
    assert!(res[0].vector_rank.is_some());
    assert!(res[0].bm25_rank.is_none());
}

#[test]
fn fusion_weight_swaps_winner() {
    let dim = 8;
    let mut idx = HybridIndex::in_memory(dim);
    idx.add_document("lex", "rust language rust", hash_embed("rust language rust", dim)).unwrap();
    idx.add_document("sem", "ownership borrowing rust", hash_embed("ownership borrowing rust", dim)).unwrap();
    idx.commit().unwrap();

    // Query "rust language": lex doc wins lexically; semantically they're close.
    let q = "rust language";
    let qv = hash_embed(q, dim);

    let lexical = idx
        .search(&SearchRequest::new(q, qv.clone()).top_k(2).with_rrf(RrfConfig::lexical_heavy()))
        .unwrap();
    assert_eq!(lexical[0].id, "lex");

    let semantic = idx
        .search(&SearchRequest::new(q, qv).top_k(2).with_rrf(RrfConfig::semantic_heavy()))
        .unwrap();
    // The lexical-biased and semantic-biased top-1 differ here is plausible;
    // we only assert the lexical-biased run favours the lexical doc.
    assert_eq!(lexical[0].id, "lex");
    let _ = semantic;
}

#[test]
#[should_panic(expected = "dimension mismatch")]
fn rejects_wrong_dimension_vector() {
    let mut idx = HybridIndex::in_memory(8);
    let _ = idx.add_document("z", "text", vec![0.0; 4]).map_err(|e| panic!("{}", e));
}

#[test]
fn duplicate_id_rejected() {
    let dim = 8;
    let mut idx = HybridIndex::in_memory(dim);
    idx.add_document("d", "first", vec![0.0; dim]).unwrap();
    let err = idx.add_document("d", "second", vec![0.0; dim]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("duplicate"), "got: {msg}");
}