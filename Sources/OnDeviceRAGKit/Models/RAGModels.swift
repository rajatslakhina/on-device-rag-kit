import Foundation

// MARK: - Documents & chunks

/// A source document handed to the pipeline for ingestion.
public struct RAGDocument: Sendable, Identifiable, Equatable {
    public let id: String
    public let title: String
    public let text: String

    public init(id: String, title: String, text: String) {
        self.id = id
        self.title = title
        self.text = text
    }
}

/// A contiguous slice of a document produced by a `ChunkingStrategy`.
///
/// Chunk IDs are deterministic (`"<documentID>#<index>"`) so that re-ingesting
/// the same document produces the same IDs, which the vector indexes use to
/// make re-ingestion idempotent (duplicate IDs are skipped, not double-indexed).
public struct Chunk: Sendable, Identifiable, Equatable {
    public let id: String
    public let documentID: String
    public let documentTitle: String
    public let text: String
    /// Zero-based position of this chunk within its document.
    public let index: Int

    public init(documentID: String, documentTitle: String, text: String, index: Int) {
        self.id = "\(documentID)#\(index)"
        self.documentID = documentID
        self.documentTitle = documentTitle
        self.text = text
        self.index = index
    }
}

/// A chunk paired with its embedding vector — the unit stored in a `VectorIndex`.
public struct EmbeddedChunk: Sendable, Identifiable, Equatable, Codable {
    public let chunk: Chunk
    public let vector: [Float]

    public var id: String { chunk.id }

    public init(chunk: Chunk, vector: [Float]) {
        self.chunk = chunk
        self.vector = vector
    }
}

extension Chunk: Codable {}

/// A chunk returned from retrieval, scored against the query.
public struct RetrievedChunk: Sendable, Identifiable, Equatable {
    public let chunk: Chunk
    public let vector: [Float]
    /// Cosine similarity to the query in [-1, 1] (0 for degenerate zero vectors).
    public let score: Float

    public var id: String { chunk.id }

    public init(chunk: Chunk, vector: [Float], score: Float) {
        self.chunk = chunk
        self.vector = vector
        self.score = score
    }
}

// MARK: - Context assembly output

/// The context block assembled from retrieved chunks under a token budget.
public struct AssembledContext: Sendable, Equatable {
    public let query: String
    /// Chunks that made it into the context window, in assembly order.
    public let includedChunks: [RetrievedChunk]
    /// Number of retrieved candidates dropped because they did not fit the budget.
    public let droppedCount: Int
    /// Estimated token count of `contextText` (heuristic — see `TokenEstimating`).
    public let estimatedTokens: Int
    /// The final concatenated context handed to the generator.
    public let contextText: String

    public init(
        query: String,
        includedChunks: [RetrievedChunk],
        droppedCount: Int,
        estimatedTokens: Int,
        contextText: String
    ) {
        self.query = query
        self.includedChunks = includedChunks
        self.droppedCount = droppedCount
        self.estimatedTokens = estimatedTokens
        self.contextText = contextText
    }
}

// MARK: - Ingestion receipt

/// Returned by `RAGPipeline.ingest` — proof of what was actually indexed.
public struct IngestReceipt: Sendable, Equatable {
    public let documentID: String
    public let chunksIndexed: Int
    /// Chunk IDs skipped because they were already present (idempotent re-ingest).
    public let duplicatesSkipped: Int

    public init(documentID: String, chunksIndexed: Int, duplicatesSkipped: Int) {
        self.documentID = documentID
        self.chunksIndexed = chunksIndexed
        self.duplicatesSkipped = duplicatesSkipped
    }
}

// MARK: - Errors

/// Every failure mode the pipeline can surface, made explicit so callers can
/// route them (retry, surface to UI, drop) instead of pattern-matching strings.
public enum RAGError: Error, Equatable, LocalizedError {
    /// The query was empty or whitespace-only.
    case emptyQuery
    /// A vector's dimension did not match the index/embedder dimension.
    case dimensionMismatch(expected: Int, got: Int)
    /// The embedder returned a different number of vectors than texts requested.
    case embeddingCountMismatch(expected: Int, got: Int)
    /// A persisted index snapshot could not be decoded.
    case snapshotCorrupted(String)
    /// A component was configured with invalid parameters.
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .emptyQuery:
            return "Query is empty or whitespace-only."
        case let .dimensionMismatch(expected, got):
            return "Vector dimension mismatch: expected \(expected), got \(got)."
        case let .embeddingCountMismatch(expected, got):
            return "Embedder returned \(got) vectors for \(expected) texts."
        case let .snapshotCorrupted(detail):
            return "Index snapshot is corrupted: \(detail)"
        case let .invalidConfiguration(detail):
            return "Invalid configuration: \(detail)"
        }
    }
}
