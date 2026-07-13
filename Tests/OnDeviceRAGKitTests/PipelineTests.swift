import XCTest
@testable import OnDeviceRAGKit

final class PipelineTests: XCTestCase {

    private func collect(_ stream: AsyncThrowingStream<String, Error>) async throws -> String {
        var output = ""
        for try await token in stream {
            output += token
        }
        return output
    }

    // MARK: - End-to-end retrieval

    func testRetrievesTheRelevantDocument() async throws {
        let pipeline = Fixtures.makePipeline()
        try await pipeline.ingest(Fixtures.swiftDoc)
        try await pipeline.ingest(Fixtures.cacheDoc)
        try await pipeline.ingest(Fixtures.coffeeDoc)

        let retrieved = try await pipeline.retrieve(query: "least recently used cache eviction")
        XCTAssertFalse(retrieved.isEmpty)
        XCTAssertEqual(retrieved.first?.chunk.documentID, "doc-cache")

        let other = try await pipeline.retrieve(query: "steaming milk for latte art microfoam")
        XCTAssertEqual(other.first?.chunk.documentID, "doc-coffee")
    }

    func testAnswerStreamCitesSources() async throws {
        let pipeline = Fixtures.makePipeline()
        try await pipeline.ingest(Fixtures.swiftDoc)

        let answer = try await pipeline.answer(query: "actor reentrancy await suspension")
        XCTAssertFalse(answer.context.includedChunks.isEmpty)
        let text = try await collect(answer.stream)
        XCTAssertTrue(text.contains("Swift Concurrency Guide"))
        XCTAssertTrue(text.contains("Based on"))
    }

    // MARK: - Defined edge states

    func testEmptyQueryThrows() async throws {
        let pipeline = Fixtures.makePipeline()
        try await pipeline.ingest(Fixtures.swiftDoc)
        do {
            _ = try await pipeline.retrieve(query: "   \n  ")
            XCTFail("expected emptyQuery")
        } catch {
            XCTAssertEqual(error as? RAGError, .emptyQuery)
        }
    }

    func testQueryAgainstEmptyIndexIsDefinedNotAnError() async throws {
        let pipeline = Fixtures.makePipeline()
        let answer = try await pipeline.answer(query: "anything at all")
        XCTAssertTrue(answer.context.includedChunks.isEmpty)
        let text = try await collect(answer.stream)
        XCTAssertTrue(text.contains("could not find"))
    }

    func testIngestingEmptyDocumentIsANoOp() async throws {
        let pipeline = Fixtures.makePipeline()
        let receipt = try await pipeline.ingest(RAGDocument(id: "empty", title: "e", text: "  \n "))
        XCTAssertEqual(receipt.chunksIndexed, 0)
        let count = await pipeline.indexedChunkCount
        XCTAssertEqual(count, 0)
    }

    // MARK: - Failure modes

    func testEmbedderFailureLeavesIndexUntouched() async throws {
        let embedder = FlakyEmbedder(failuresBeforeSuccess: Int.max)
        let pipeline = Fixtures.makePipeline(embedder: embedder)
        do {
            try await pipeline.ingest(Fixtures.swiftDoc)
            XCTFail("expected embedder failure")
        } catch {
            XCTAssertTrue(error is TestError)
        }
        let count = await pipeline.indexedChunkCount
        XCTAssertEqual(count, 0, "a failed ingest must index nothing (per-document atomicity)")
    }

    func testContractViolatingEmbedderIsCaughtNotCrashed() async throws {
        let pipeline = Fixtures.makePipeline(embedder: ShortChangingEmbedder())
        do {
            try await pipeline.ingest(Fixtures.swiftDoc)
            XCTFail("expected embeddingCountMismatch")
        } catch {
            guard case .embeddingCountMismatch = error as? RAGError else {
                return XCTFail("expected embeddingCountMismatch, got \(error)")
            }
        }
        let count = await pipeline.indexedChunkCount
        XCTAssertEqual(count, 0)
    }

    func testMixedDimensionBatchLeavesIndexUntouched() async throws {
        // An embedder returning correct count but MIXED dimensions within one
        // batch must not leave a partial document behind: index `add` validates
        // the whole batch before mutating (atomic add).
        let pipeline = Fixtures.makePipeline(embedder: MixedDimensionEmbedder())
        do {
            try await pipeline.ingest(Fixtures.swiftDoc)
            XCTFail("expected dimensionMismatch")
        } catch {
            guard case .dimensionMismatch = error as? RAGError else {
                return XCTFail("expected dimensionMismatch, got \(error)")
            }
        }
        let count = await pipeline.indexedChunkCount
        XCTAssertEqual(count, 0, "atomic add: nothing from the mixed batch may be inserted")
    }

    func testMidStreamGeneratorFailureSurfacesThroughTheStream() async throws {
        let pipeline = Fixtures.makePipeline(generator: MidStreamFailingGenerator())
        try await pipeline.ingest(Fixtures.swiftDoc)
        let answer = try await pipeline.answer(query: "actors")
        do {
            _ = try await collect(answer.stream)
            XCTFail("expected mid-stream failure")
        } catch {
            XCTAssertTrue(error is TestError)
        }
    }

    // MARK: - Idempotency

    func testReingestingSameDocumentSkipsDuplicates() async throws {
        let pipeline = Fixtures.makePipeline()
        let first = try await pipeline.ingest(Fixtures.cacheDoc)
        XCTAssertGreaterThan(first.chunksIndexed, 0)
        XCTAssertEqual(first.duplicatesSkipped, 0)

        let second = try await pipeline.ingest(Fixtures.cacheDoc)
        XCTAssertEqual(second.chunksIndexed, 0)
        XCTAssertEqual(second.duplicatesSkipped, first.chunksIndexed)

        let count = await pipeline.indexedChunkCount
        XCTAssertEqual(count, first.chunksIndexed)
    }

    // MARK: - Persistence

    private func temporaryFileURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ragkit-pipeline-\(UUID().uuidString).json")
    }

    func testSnapshotRoundTripRestoresRetrieval() async throws {
        let url = temporaryFileURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let original = Fixtures.makePipeline()
        try await original.ingest(Fixtures.cacheDoc)
        try await original.ingest(Fixtures.coffeeDoc)
        try await original.saveSnapshot(to: url)

        let restored = Fixtures.makePipeline()
        let inserted = try await restored.restoreSnapshot(from: url)
        XCTAssertGreaterThan(inserted, 0)

        let retrieved = try await restored.retrieve(query: "LRU eviction capacity")
        XCTAssertEqual(retrieved.first?.chunk.documentID, "doc-cache")
    }

    func testCorruptedSnapshotRestoreLeavesIndexUntouched() async throws {
        let url = temporaryFileURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("garbage!!!".utf8).write(to: url)

        let pipeline = Fixtures.makePipeline()
        try await pipeline.ingest(Fixtures.swiftDoc)
        let before = await pipeline.indexedChunkCount

        do {
            _ = try await pipeline.restoreSnapshot(from: url)
            XCTFail("expected snapshotCorrupted")
        } catch {
            guard case .snapshotCorrupted = error as? RAGError else {
                return XCTFail("expected snapshotCorrupted, got \(error)")
            }
        }
        let after = await pipeline.indexedChunkCount
        XCTAssertEqual(before, after, "failed restore must not partially insert")
    }

    // MARK: - Concurrency

    func testConcurrentIngestAndQueryReachCompleteConsistentState() async throws {
        let pipeline = Fixtures.makePipeline(
            embedder: CachingEmbedder(wrapping: DeterministicHashEmbedder(dimension: 64))
        )
        let docs = [Fixtures.swiftDoc, Fixtures.cacheDoc, Fixtures.coffeeDoc]

        try await withThrowingTaskGroup(of: Void.self) { group in
            for doc in docs {
                group.addTask {
                    try await pipeline.ingest(doc)
                }
            }
            // Interleave queries mid-ingest: they may see any prefix of the
            // corpus, but must never throw or observe a partial document.
            for _ in 0..<5 {
                group.addTask {
                    _ = try await pipeline.retrieve(query: "cache eviction")
                }
            }
            try await group.waitForAll()
        }

        // End state: everything indexed exactly once.
        let expected = try await countChunks(in: docs)
        let count = await pipeline.indexedChunkCount
        XCTAssertEqual(count, expected)

        let retrieved = try await pipeline.retrieve(query: "least recently used cache eviction")
        XCTAssertEqual(retrieved.first?.chunk.documentID, "doc-cache")
    }

    func testWorksWithPartitionedIndexEndToEnd() async throws {
        let pipeline = Fixtures.makePipeline(
            index: PartitionedVectorIndex(dimension: 64, centroidCount: 4, probeCount: 2)
        )
        try await pipeline.ingest(Fixtures.swiftDoc)
        try await pipeline.ingest(Fixtures.cacheDoc)
        try await pipeline.ingest(Fixtures.coffeeDoc)

        let retrieved = try await pipeline.retrieve(query: "microfoam latte art milk")
        XCTAssertFalse(retrieved.isEmpty)
        XCTAssertEqual(retrieved.first?.chunk.documentID, "doc-coffee")
    }

    // MARK: - Helpers

    private func countChunks(in docs: [RAGDocument]) async throws -> Int {
        let chunker = ParagraphChunker(targetSize: 200)
        return docs.reduce(0) { $0 + chunker.chunk($1).count }
    }
}
