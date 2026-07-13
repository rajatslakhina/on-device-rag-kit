import Foundation
@testable import OnDeviceRAGKit

struct TestError: Error, Equatable {}

/// Counts calls/texts through to a real deterministic embedder.
/// Used to prove caching and coalescing behavior rather than assert it.
actor CountingEmbedder: Embedder {
    let dimension: Int
    private let inner: DeterministicHashEmbedder
    private let delayNanoseconds: UInt64
    private(set) var callCount = 0
    private(set) var textCount = 0

    init(dimension: Int = 64, delayNanoseconds: UInt64 = 0) {
        self.dimension = dimension
        self.inner = DeterministicHashEmbedder(dimension: dimension)
        self.delayNanoseconds = delayNanoseconds
    }

    func embed(_ texts: [String]) async throws -> [[Float]] {
        callCount += 1
        textCount += texts.count
        if delayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: delayNanoseconds)
        }
        return try await inner.embed(texts)
    }
}

/// Fails the first `failuresBeforeSuccess` calls, then succeeds.
/// `failuresBeforeSuccess = Int.max` fails forever.
actor FlakyEmbedder: Embedder {
    let dimension: Int
    private let inner: DeterministicHashEmbedder
    private var remainingFailures: Int
    private(set) var callCount = 0

    init(dimension: Int = 64, failuresBeforeSuccess: Int = Int.max) {
        self.dimension = dimension
        self.inner = DeterministicHashEmbedder(dimension: dimension)
        self.remainingFailures = failuresBeforeSuccess
    }

    func embed(_ texts: [String]) async throws -> [[Float]] {
        callCount += 1
        if remainingFailures > 0 {
            remainingFailures -= 1
            throw TestError()
        }
        return try await inner.embed(texts)
    }
}

/// Violates the Embedder contract by returning one fewer vector than asked —
/// exists to prove the pipeline validates instead of crashing on zip misalignment.
struct ShortChangingEmbedder: Embedder {
    let dimension: Int = 64

    func embed(_ texts: [String]) async throws -> [[Float]] {
        let inner = DeterministicHashEmbedder(dimension: dimension)
        let all = try await inner.embed(texts)
        return Array(all.dropLast())
    }
}

/// Violates the Embedder contract differently: returns the right COUNT of
/// vectors but mixes dimensions within one batch — exists to prove `add` is
/// atomic (validate-all-then-insert) rather than mutate-as-you-validate.
struct MixedDimensionEmbedder: Embedder {
    let dimension: Int = 64

    func embed(_ texts: [String]) async throws -> [[Float]] {
        let inner = DeterministicHashEmbedder(dimension: dimension)
        var all = try await inner.embed(texts)
        if !all.isEmpty {
            all[all.count - 1] = [1.0, 0.0] // wrong dimension, last position
        }
        return all
    }
}

/// Generator whose stream yields a couple of tokens and then fails —
/// exercises the mid-stream failure path a real model backend can hit.
struct MidStreamFailingGenerator: Generator {
    func generate(context: AssembledContext) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield("partial")
            continuation.yield(" output")
            continuation.finish(throwing: TestError())
        }
    }
}

enum Fixtures {
    static let swiftDoc = RAGDocument(
        id: "doc-swift",
        title: "Swift Concurrency Guide",
        text: """
        Swift actors protect mutable state by serializing access through an actor mailbox. \
        Reentrancy means an actor can interleave other work at every await suspension point.

        Structured concurrency ties child task lifetimes to their parent scope. \
        Task groups fan out work and rejoin results deterministically.
        """
    )

    static let cacheDoc = RAGDocument(
        id: "doc-cache",
        title: "Cache Design Notes",
        text: """
        An LRU cache evicts the least recently used entry when capacity is exceeded. \
        Cost-based eviction counts bytes instead of entries.

        Disk tiers persist entries across relaunch using a manifest of hashed filenames. \
        Memory pressure should trim the cache fractionally rather than flushing everything.
        """
    )

    static let coffeeDoc = RAGDocument(
        id: "doc-coffee",
        title: "Espresso Handbook",
        text: """
        Espresso extraction balances grind size, dose, and contact time. \
        A finer grind slows the shot and increases extraction yield.

        Milk steaming builds microfoam by stretching then rolling the milk. \
        Latte art requires velvety texture rather than stiff foam.
        """
    )

    static func makePipeline(
        index: (any VectorIndex)? = nil,
        embedder: (any Embedder)? = nil,
        generator: any Generator = SimulatedGenerator(),
        configuration: RAGPipeline.Configuration = .init()
    ) -> RAGPipeline {
        let dimension = embedder?.dimension ?? 64
        return RAGPipeline(
            embedder: embedder ?? DeterministicHashEmbedder(dimension: dimension),
            index: index ?? LinearVectorIndex(dimension: dimension),
            chunker: ParagraphChunker(targetSize: 200),
            reranker: MMRReranker(lambda: 0.7),
            assembler: ContextAssembler(tokenBudget: 512),
            generator: generator,
            configuration: configuration
        )
    }

    /// Deterministic synthetic embedded chunk for index-level tests.
    static func entry(id docID: String, index: Int, vector: [Float]) -> EmbeddedChunk {
        EmbeddedChunk(
            chunk: Chunk(documentID: docID, documentTitle: docID, text: "chunk \(docID) \(index)", index: index),
            vector: vector
        )
    }
}
