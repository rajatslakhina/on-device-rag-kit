# OnDeviceRAGKit

**A complete on-device RAG retrieval system for iOS — chunking, embedding, ANN vector search, MMR reranking, token-budgeted context assembly, and streaming generation — with every stage behind a protocol seam and every failure mode a defined, tested state.**

Most "RAG on iOS" samples are a network call wearing a trench coat: send text to an embeddings API, cosine-sort an array, paste into a prompt. That works until the user is offline, the corpus hits five figures, two features embed the same text simultaneously, or the app relaunches and re-embeds everything it already paid battery for. This library is the other 90% — the *system* around the model.

## Why this matters

Apple opened the Foundation Models framework to any LLM provider at WWDC26, and Spotlight-style on-device RAG is now a first-party pattern. The architectural question an iOS engineering lead has to answer is no longer *"can I run retrieval on device?"* but *"which parts of the retrieval stack are commitments, and which are swappable?"* This repo is that answer as running code:

- **Swappable:** the embedder (`NLContextualEmbedding` today, MLX tomorrow, cloud API on Wi-Fi), the index (exact vs. ANN), the ranking policy, the tokenizer, the generator. Each is a small protocol.
- **Commitments:** deterministic retrieval, per-document ingest atomicity, defined behavior for every degenerate input, and embeddings that survive relaunch.

```
ingest:  RAGDocument ─▶ ChunkingStrategy ─▶ Embedder ─▶ VectorIndex
query:   text ─▶ Embedder ─▶ VectorIndex.search ─▶ Reranker ─▶ ContextAssembler ─▶ Generator
                                   │                                                  │
                        IndexSnapshotStore (relaunch persistence)          AsyncThrowingStream<String>
```

## Design decisions (and the alternatives they beat)

**One actor owns the pipeline; the index is not its own actor.**
`RAGPipeline` is the single concurrency boundary. *Rejected:* per-component actors — every query would double-hop between isolation domains, adding latency and reintroducing ordering questions between hops, for state that is never shared outside the pipeline anyway.

**Per-document ingest atomicity, stated precisely.**
Actors are re-entrant at `await`. The pipeline stages chunks locally across the embedding suspension point and commits to the index *synchronously* — so a document is fully indexed or not at all, and queries never observe a half-added document. Cross-document completion order under concurrent ingestion is explicitly *not* guaranteed; serializing it would forfeit embedder batching for a guarantee most callers don't need. That trade-off is documented in the type, where a reviewer would look for it.

**Exact search is the default; ANN is the opt-in.**
`LinearVectorIndex` gives perfect recall at O(n·d) — a few million multiply-adds at 50k chunks, fine on any A-series chip. `PartitionedVectorIndex` is an IVF-style index with online centroid updates for streaming ingestion. *Rejected:* HNSW — a better frontier at web scale, but its per-entry graph memory and implementation complexity are indefensible at on-device corpus sizes. *Rejected:* offline k-means training — higher-quality centroids, but incompatible with streaming ingestion.

**MMR reranking, not raw top-k.**
Overlapping chunkers guarantee that raw top-k is often the same passage k times; MMR trades a little relevance for coverage. *Rejected:* cross-encoder reranking — strictly better quality, but costs a model forward-pass per candidate, which is exactly the battery/latency budget this pipeline protects.

**Coalescing LRU embed-cache with fractional trim.**
`CachingEmbedder` closes the actor-reentrancy gap where two concurrent requests for the same text would both miss cache and embed twice (an in-flight batch registry). Failure semantics are stated precisely: the first caller to observe a batch failure synchronously evicts the *entire batch* — every sibling text, not just its own — so a transient failure costs at most one failed call per batch and the next request re-embeds fresh (with the standard single-flight caveat that a joining caller receives the joined computation's outcome). Memory pressure calls `trim(toFraction:)`. *Rejected:* `removeAll()` on memory warning — it zeroes the hit rate at the exact moment the system can least afford re-embedding everything.

**Deterministic everything.**
FNV-1a feature hashing (never Swift's randomly-seeded `hashValue`, which would silently corrupt persisted indexes across launches), deterministic chunk IDs (`docID#index`) that make re-ingestion idempotent, and score-then-ID tie-breaks so identical corpora always retrieve identically — the property that makes retrieval bugs reproducible.

**Versioned JSON snapshots.**
Embeddings are the expensive artifact; throwing them away between launches wastes battery. Snapshots are atomic writes with version + dimension validation that fail loudly (`snapshotCorrupted`) instead of misdecoding. *Rejected:* mmap'd binary — faster at a scale this library explicitly does not target, unauditable in review, fragile across versions.

**The bundled embedder/generator are deliberate simulations.**
`DeterministicHashEmbedder` is a bag-of-features lexical embedder; `SimulatedGenerator` streams a grounded, source-citing template. They exist so the *system* — retrieval, ranking, budgeting, persistence, failure paths — is fully exercised and CI-testable on Linux. Real backends (`NLContextualEmbedding`, MLX, Foundation Models `LanguageModelSession`) plug into `Embedder`/`Generator` without touching anything else. This boundary is the honest one: the library's product is the retrieval system, not the model.

## Failure modes with defined, tested behavior

| Failure | Behavior |
|---|---|
| Embedder throws mid-document | Nothing indexed (atomic ingest); index count unchanged |
| Embedder returns wrong vector count | `embeddingCountMismatch` thrown — validated, never zip-misaligned |
| Query on empty index | Defined non-error state: empty context, honest "nothing indexed" answer |
| Empty/whitespace query | `RAGError.emptyQuery` |
| Zero vector (empty text) | Cosine similarity 0 — no NaN, no divide-by-zero |
| Corrupted/mismatched snapshot | `snapshotCorrupted`; index left untouched |
| Concurrent same-text embeds | Coalesced to one backend call |
| Mid-stream generation failure | Error surfaces through the stream, retrieval state unaffected |
| Overlap ≥ chunk size, k ≤ 0, budget ≤ 0, λ outside [0,1] | Clamped or defined-empty — never a trap or infinite loop |

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/rajatslakhina/on-device-rag-kit.git", branch: "main")
]
```

## Usage

```swift
import OnDeviceRAGKit

let embedder = CachingEmbedder(wrapping: DeterministicHashEmbedder(dimension: 128))
let pipeline = RAGPipeline(
    embedder: embedder,
    index: LinearVectorIndex(dimension: 128)
)

try await pipeline.ingest(RAGDocument(id: "notes", title: "Team Notes", text: longText))

let answer = try await pipeline.answer(query: "what did we decide about caching?")
for try await token in answer.stream {
    print(token, terminator: "")
}
// answer.context carries full retrieval provenance: chunks, scores, budget usage.
```

## Tests

`swift test` — 77 tests covering the crash-and-correctness edges above: chunker termination under degenerate configs, LRU eviction order, coalescing under concurrency, failure-non-poisoning, ANN/exact top-hit parity on separated clusters, ingest atomicity, snapshot corruption, mid-stream failure, and concurrent ingest+query consistency.

## Verification (honest)

- `swift build` and `swift test` (77/77 passing) were run on a Linux Swift toolchain — the library is pure Foundation by design, so this exercises all of it.
- The companion demo app is a separate repo consuming this package as a **remote** SPM dependency by its git URL, exactly like an external consumer.

## Demo app

→ **[on-device-rag-kit-demo-app](https://github.com/rajatslakhina/on-device-rag-kit-demo-app)** — a SwiftUI app that consumes this package as a **remote** SPM dependency (by this repo's git URL, branch `main`, exactly like an external consumer) and puts the pipeline on screen: streaming answers with full retrieval provenance, a live exact-vs-ANN index swap absorbed by the embed cache, and a memory-pressure trim button wired to `trim(toFraction:)`.

## License

MIT
