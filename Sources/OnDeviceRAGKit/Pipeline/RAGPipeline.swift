import Foundation

/// The answer to a query: what the generator was given, plus the live stream.
public struct RAGAnswer: Sendable {
    /// Full retrieval provenance — which chunks, at what similarity, under
    /// what budget. Surfaced so UIs can show *why* an answer says what it says
    /// (retrieval debuggability is the difference between a demo and a system).
    public let context: AssembledContext
    public let stream: AsyncThrowingStream<String, Error>

    public init(context: AssembledContext, stream: AsyncThrowingStream<String, Error>) {
        self.context = context
        self.stream = stream
    }
}

/// Orchestrates the full on-device RAG flow:
///
///     ingest:  document ─▶ chunker ─▶ embedder ─▶ vector index
///     query:   text ─▶ embedder ─▶ index.search ─▶ reranker ─▶ assembler ─▶ generator
///
/// **Concurrency model.** The pipeline is an actor and is the *single*
/// concurrency boundary: chunker/reranker/assembler are value types, the
/// embedder manages its own isolation, and the index is exclusively owned by
/// this actor from construction onward (see `VectorIndex` docs for why the
/// index is not its own actor).
///
/// **Ordering guarantees, stated precisely.** Actors are re-entrant: at the
/// `await embedder.embed(...)` suspension point, other ingest/query calls may
/// interleave. What this design guarantees:
/// - *Per-document ingest atomicity* — chunks are staged locally and the index
///   `add` happens synchronously (no suspension) after embedding succeeds, so
///   a document is either fully indexed or not indexed at all, regardless of
///   interleaving. An embedder failure indexes nothing.
/// - *Queries never observe partial documents* — `index.search` runs
///   synchronously between suspension points, so it sees the index either
///   before or after any given document's atomic `add`, never mid-add.
/// - *Cross-document completion order is NOT guaranteed* to match submission
///   order when callers ingest concurrently — callers needing strict corpus
///   ordering should `await` each ingest before the next. This is documented
///   rather than "fixed" because serializing embedding batches end-to-end
///   would forfeit embedder-level batching for a guarantee most callers
///   don't need.
public actor RAGPipeline {
    public struct Configuration: Sendable {
        /// Candidates fetched from the index before reranking. Fetching more
        /// than the final context limit gives the reranker room to trade
        /// relevance for diversity.
        public var retrievalK: Int
        /// Maximum chunks forwarded to context assembly after reranking.
        public var contextChunkLimit: Int

        public init(retrievalK: Int = 12, contextChunkLimit: Int = 5) {
            self.retrievalK = max(1, retrievalK)
            self.contextChunkLimit = max(1, contextChunkLimit)
        }
    }

    private let embedder: any Embedder
    private let index: any VectorIndex
    private let chunker: any ChunkingStrategy
    private let reranker: any Reranker
    private let assembler: ContextAssembler
    private let generator: any Generator
    private let configuration: Configuration

    public init(
        embedder: any Embedder,
        index: any VectorIndex,
        chunker: any ChunkingStrategy = ParagraphChunker(),
        reranker: any Reranker = MMRReranker(),
        assembler: ContextAssembler = ContextAssembler(),
        generator: any Generator = SimulatedGenerator(),
        configuration: Configuration = Configuration()
    ) {
        self.embedder = embedder
        self.index = index
        self.chunker = chunker
        self.reranker = reranker
        self.assembler = assembler
        self.generator = generator
        self.configuration = configuration
    }

    // MARK: - Ingestion

    /// Chunks, embeds, and indexes one document — atomically (see type docs).
    @discardableResult
    public func ingest(_ document: RAGDocument) async throws -> IngestReceipt {
        let chunks = chunker.chunk(document)
        guard !chunks.isEmpty else {
            return IngestReceipt(documentID: document.id, chunksIndexed: 0, duplicatesSkipped: 0)
        }

        // Stage locally; nothing touches the index until embedding succeeded
        // for the WHOLE document.
        let vectors = try await embedder.embed(chunks.map { $0.text })
        guard vectors.count == chunks.count else {
            throw RAGError.embeddingCountMismatch(expected: chunks.count, got: vectors.count)
        }

        var staged: [EmbeddedChunk] = []
        staged.reserveCapacity(chunks.count)
        for (offset, chunk) in chunks.enumerated() {
            // zip-style pairing with an explicit bounds guard instead of
            // trusting the count check above forever.
            guard offset < vectors.count else {
                throw RAGError.embeddingCountMismatch(expected: chunks.count, got: vectors.count)
            }
            staged.append(EmbeddedChunk(chunk: chunk, vector: vectors[offset]))
        }

        let inserted = try index.add(staged) // synchronous — atomic commit point
        return IngestReceipt(
            documentID: document.id,
            chunksIndexed: inserted,
            duplicatesSkipped: staged.count - inserted
        )
    }

    // MARK: - Retrieval

    /// Embeds the query, searches the index, and reranks.
    /// - Throws: `RAGError.emptyQuery` for empty/whitespace queries.
    public func retrieve(query: String) async throws -> [RetrievedChunk] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw RAGError.emptyQuery }

        let vectors = try await embedder.embed([trimmed])
        guard let queryVector = vectors.first else {
            throw RAGError.embeddingCountMismatch(expected: 1, got: 0)
        }

        let candidates = try index.search(query: queryVector, k: configuration.retrievalK)
        return reranker.rerank(
            query: queryVector,
            candidates: candidates,
            limit: configuration.contextChunkLimit
        )
    }

    /// Full query path: retrieve → assemble → generate.
    ///
    /// Querying an *empty index* is a defined, non-throwing state: the
    /// assembled context is empty and the generator produces an honest
    /// "nothing indexed matches" answer — a first-launch UI state, not an error.
    public func answer(query: String) async throws -> RAGAnswer {
        let retrieved = try await retrieve(query: query)
        let context = assembler.assemble(query: query, candidates: retrieved)
        let stream = generator.generate(context: context)
        return RAGAnswer(context: context, stream: stream)
    }

    // MARK: - Introspection & lifecycle

    public var indexedChunkCount: Int { index.count }

    public func reset() {
        index.removeAll()
    }

    // MARK: - Persistence

    /// Persists the current index contents atomically to `url`.
    public func saveSnapshot(to url: URL) throws {
        let store = IndexSnapshotStore(fileURL: url)
        try store.save(entries: index.entries, dimension: index.dimension)
    }

    /// Restores entries from a snapshot into the index (idempotent thanks to
    /// chunk-ID dedupe). Returns the number of entries actually inserted.
    /// - Throws: `RAGError.snapshotCorrupted` — and on ANY throw the index is
    ///   left untouched: validation happens inside `load` before insertion.
    @discardableResult
    public func restoreSnapshot(from url: URL) throws -> Int {
        let store = IndexSnapshotStore(fileURL: url)
        let entries = try store.load(expectedDimension: index.dimension)
        return try index.add(entries)
    }
}
