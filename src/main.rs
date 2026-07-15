use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use hybrid_search::{hash_embed, HybridIndex, SearchRequest};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

/// Hybrid search index (BM25 + vector + RRF).
#[derive(Parser, Debug)]
#[command(name = "hybrid-search", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,

    /// Vector dimensionality expected by the index.
    #[arg(long, default_value_t = 256, global = true)]
    dim: usize,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the built-in demo on a tiny synthetic corpus.
    Demo,
    /// Build an index from a JSONL file: one `{"id","text","vector"}` per line.
    /// If `vector` is absent it is synthesized with a deterministic hashing embed.
    Index {
        /// Input JSONL file.
        input: PathBuf,
        /// Output directory for the saved index (currently kept in memory;
        /// the path is reserved for a future persistent backend).
        #[arg(long, default_value = "index.out")]
        out: PathBuf,
    },
    /// Search an index built from a JSONL corpus.
    Search {
        /// The JSONL corpus (re-indexed in memory — fast for small/medium sets).
        corpus: PathBuf,
        /// Query string (keyword). Pass `--vector FILE` to also do semantic.
        query: Option<String>,
        /// Optional path to a JSON file containing the query vector as a float array.
        #[arg(long)]
        vector: Option<PathBuf>,
        /// Top-K results.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// RRF k constant.
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f32,
        /// Weight on the BM25 signal in [0,1] (vector weight = 1 - bm25_weight).
        #[arg(long, default_value_t = 0.5)]
        bm25_weight: f32,
        /// If set and no query vector is provided, synthesize one from the query
        /// text using the deterministic hashing embed.
        #[arg(long)]
        auto_embed: bool,
        /// Print results as JSON.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    // Keep tantivy's commit logs quiet by default; override with RUST_LOG.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Demo => run_demo(cli.dim),
        Cmd::Index { input, out: _ } => {
            let mut idx = HybridIndex::in_memory(cli.dim);
            load_jsonl(&mut idx, &input, cli.dim)?;
            idx.commit().map_err(|e| anyhow!(e.to_string()))?;
            println!("Indexed {} documents (dim={}).", idx.len(), cli.dim);
            Ok(())
        }
        Cmd::Search {
            corpus,
            query,
            vector,
            top_k,
            rrf_k,
            bm25_weight,
            auto_embed,
            json,
        } => {
            let mut idx = HybridIndex::in_memory(cli.dim);
            load_jsonl(&mut idx, &corpus, cli.dim)?;
            idx.commit().map_err(|e| anyhow!(e.to_string()))?;

            let qvec = if let Some(path) = vector {
                Some(read_vector(&path)?)
            } else if auto_embed {
                query.as_ref().map(|q| hash_embed(q, cli.dim))
            } else {
                None
            };

            let qtext = query;

            if qtext.is_none() && qvec.is_none() {
                return Err(anyhow!("provide a query string and/or --vector/--auto-embed"));
            }

            let req = build_request(qtext, qvec, top_k, rrf_k, bm25_weight);
            let start = Instant::now();
            let results = idx.search(&req).map_err(|e| anyhow!(e.to_string()))?;
            let elapsed = start.elapsed();

            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                println!("Top-{} results (hybrid, {:.2?} ms):", top_k, elapsed.as_secs_f32() * 1000.0);
                for (i, r) in results.iter().enumerate() {
                    println!(
                        "{:2}. [score={:.4}] id={}\n     bm25={:?} vec={:?}\n     text: {}",
                        i + 1,
                        r.score,
                        r.id,
                        r.bm25_rank,
                        r.vector_rank,
                        r.text.chars().take(120).collect::<String>()
                    );
                }
            }
            Ok(())
        }
    }
}

fn build_request(
    qtext: Option<String>,
    qvec: Option<Vec<f32>>,
    top_k: usize,
    rrf_k: f32,
    bm25_weight: f32,
) -> SearchRequest {
    let mut req = if let (Some(t), Some(v)) = (qtext.clone(), qvec.clone()) {
        SearchRequest::new(t, v)
    } else if let Some(t) = qtext {
        SearchRequest::text(t)
    } else if let Some(v) = qvec {
        SearchRequest::vector(v)
    } else {
        unreachable!()
    };
    req.top_k = top_k;
    req.rrf.k = rrf_k;
    req.rrf.bm25_weight = bm25_weight;
    req
}

fn load_jsonl(idx: &mut HybridIndex, path: &PathBuf, dim: usize) -> Result<()> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut n = 0;
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("parse JSON at {}:{}", path.display(), lineno + 1))?;

        let id = v["id"]
            .as_str()
            .ok_or_else(|| anyhow!("missing 'id' at {}:{}", path.display(), lineno + 1))?
            .to_string();
        let text = v["text"]
            .as_str()
            .ok_or_else(|| anyhow!("missing 'text' at {}:{}", path.display(), lineno + 1))?
            .to_string();
        let vector: Vec<f32> = match v.get("vector") {
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .map(|x| x.as_f64().map(|f| f as f32).unwrap_or(0.0))
                .collect(),
            _ => hash_embed(&text, dim),
        };
        idx.add_document(id, text, vector).map_err(|e| anyhow!(e.to_string()))?;
        n += 1;
    }
    tracing::info!(n, "loaded documents");
    Ok(())
}

fn read_vector(path: &PathBuf) -> Result<Vec<f32>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let v: Vec<f32> = if s.trim_start().starts_with('[') {
        serde_json::from_str(&s)?
    } else {
        s.split_whitespace().map(|t| t.parse::<f32>().unwrap_or(0.0)).collect()
    };
    Ok(v)
}

fn run_demo(dim: usize) -> Result<()> {
    let corpus = vec![
        ("d1", "The Rust programming language is fast and memory safe."),
        ("d2", "Iron and steel can rust when left out in the rain."),
        ("d3", "Memory safety prevents many classes of bugs in systems code."),
        ("d4", "A car engine burns fuel to generate mechanical power."),
        ("d5", "Ownership and borrowing are core concepts in Rust."),
        ("d6", "Borrowing money from a bank requires a good credit score."),
        ("d7", "Cargo is the package manager and build tool for Rust."),
        ("d8", "Ships and boats are also called vessels or watercraft."),
    ];

    let mut idx = HybridIndex::in_memory(dim);
    for (id, text) in &corpus {
        let vec = hash_embed(text, dim);
        idx.add_document(*id, *text, vec).map_err(|e| anyhow!(e.to_string()))?;
    }
    idx.commit().map_err(|e| anyhow!(e.to_string()))?;
    println!("Indexed {} documents (dim={}).\n", idx.len(), dim);

    for q in [
        "rust language memory safety",
        "rust on metal",
        "borrowing money",
        "what is cargo",
    ] {
        let qvec = hash_embed(q, dim);
        let req = SearchRequest::new(q, qvec).top_k(3);
        let start = Instant::now();
        let results = idx.search(&req).map_err(|e| anyhow!(e.to_string()))?;
        let ms = start.elapsed().as_secs_f32() * 1000.0;
        println!("query: {:?}  ({:.3} ms)", q, ms);
        for r in &results {
            println!(
                "  [{:.4}] {:<3} bm25_rank={:?} vec_rank={:?}  {}",
                r.score, r.id, r.bm25_rank, r.vector_rank,
                r.text.chars().take(70).collect::<String>()
            );
        }
        println!();
    }
    Ok(())
}