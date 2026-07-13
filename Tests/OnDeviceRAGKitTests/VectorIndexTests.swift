import XCTest
@testable import OnDeviceRAGKit

final class VectorIndexTests: XCTestCase {

    // Orthonormal-ish fixtures in 4 dimensions.
    private let e0: [Float] = [1, 0, 0, 0]
    private let e1: [Float] = [0, 1, 0, 0]
    private let e2: [Float] = [0, 0, 1, 0]
    private let e3: [Float] = [0, 0, 0, 1]

    // MARK: - LinearVectorIndex

    func testSearchOnEmptyIndexReturnsEmpty() throws {
        let index = LinearVectorIndex(dimension: 4)
        XCTAssertTrue(try index.search(query: e0, k: 5).isEmpty)
    }

    func testKZeroOrNegativeReturnsEmpty() throws {
        let index = LinearVectorIndex(dimension: 4)
        try index.add([Fixtures.entry(id: "a", index: 0, vector: e0)])
        XCTAssertTrue(try index.search(query: e0, k: 0).isEmpty)
        XCTAssertTrue(try index.search(query: e0, k: -3).isEmpty)
    }

    func testKLargerThanCountClamps() throws {
        let index = LinearVectorIndex(dimension: 4)
        try index.add([
            Fixtures.entry(id: "a", index: 0, vector: e0),
            Fixtures.entry(id: "b", index: 0, vector: e1),
        ])
        XCTAssertEqual(try index.search(query: e0, k: 99).count, 2)
    }

    func testAddDimensionMismatchThrows() {
        let index = LinearVectorIndex(dimension: 4)
        XCTAssertThrowsError(
            try index.add([Fixtures.entry(id: "a", index: 0, vector: [1, 0])])
        ) { error in
            XCTAssertEqual(error as? RAGError, .dimensionMismatch(expected: 4, got: 2))
        }
    }

    func testSearchDimensionMismatchThrows() {
        let index = LinearVectorIndex(dimension: 4)
        XCTAssertThrowsError(try index.search(query: [1, 0], k: 1)) { error in
            XCTAssertEqual(error as? RAGError, .dimensionMismatch(expected: 4, got: 2))
        }
    }

    func testDuplicateChunkIDsAreSkipped() throws {
        let index = LinearVectorIndex(dimension: 4)
        let entry = Fixtures.entry(id: "a", index: 0, vector: e0)
        let firstInsert = try index.add([entry])
        let secondInsert = try index.add([entry])
        XCTAssertEqual(firstInsert, 1)
        XCTAssertEqual(secondInsert, 0)
        XCTAssertEqual(index.count, 1)
    }

    func testNearestNeighborOrdering() throws {
        let index = LinearVectorIndex(dimension: 4)
        try index.add([
            Fixtures.entry(id: "exact", index: 0, vector: e0),
            Fixtures.entry(id: "close", index: 0, vector: VectorMath.normalized([0.9, 0.4, 0, 0])),
            Fixtures.entry(id: "far", index: 0, vector: e2),
        ])
        let results = try index.search(query: e0, k: 3)
        XCTAssertEqual(results.map { $0.chunk.documentID }, ["exact", "close", "far"])
    }

    func testZeroVectorQueryIsSafe() throws {
        let index = LinearVectorIndex(dimension: 4)
        try index.add([Fixtures.entry(id: "a", index: 0, vector: e0)])
        let results = try index.search(query: [0, 0, 0, 0], k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].score, 0)
    }

    func testDeterministicTieBreakByChunkID() throws {
        let index = LinearVectorIndex(dimension: 4)
        try index.add([
            Fixtures.entry(id: "b", index: 0, vector: e1),
            Fixtures.entry(id: "a", index: 0, vector: e1),
        ])
        let results = try index.search(query: e1, k: 2)
        XCTAssertEqual(results.map { $0.chunk.documentID }, ["a", "b"])
    }

    func testRemoveAllAllowsReinsertion() throws {
        let index = LinearVectorIndex(dimension: 4)
        let entry = Fixtures.entry(id: "a", index: 0, vector: e0)
        try index.add([entry])
        index.removeAll()
        XCTAssertEqual(index.count, 0)
        XCTAssertEqual(try index.add([entry]), 1)
    }

    // MARK: - PartitionedVectorIndex

    func testPartitionedBehavesExactlyBelowCentroidCount() throws {
        // With fewer entries than centroids, every entry is its own partition
        // and probing must still find the true nearest neighbor.
        let index = PartitionedVectorIndex(dimension: 4, centroidCount: 8, probeCount: 2)
        try index.add([
            Fixtures.entry(id: "a", index: 0, vector: e0),
            Fixtures.entry(id: "b", index: 0, vector: e1),
            Fixtures.entry(id: "c", index: 0, vector: e2),
        ])
        let results = try index.search(query: e1, k: 1)
        XCTAssertEqual(results.first?.chunk.documentID, "b")
    }

    func testPartitionedTopHitMatchesLinearOnSeparatedClusters() throws {
        // Two well-separated clusters; ANN must agree with exact search on
        // the top hit when the query sits inside one cluster.
        let linear = LinearVectorIndex(dimension: 4)
        let partitioned = PartitionedVectorIndex(dimension: 4, centroidCount: 2, probeCount: 1)

        var entries: [EmbeddedChunk] = []
        for i in 0..<10 {
            let jitter = Float(i) * 0.01
            entries.append(Fixtures.entry(
                id: "clusterA-\(i)", index: i,
                vector: VectorMath.normalized([1, jitter, 0, 0])
            ))
            entries.append(Fixtures.entry(
                id: "clusterB-\(i)", index: i,
                vector: VectorMath.normalized([0, 0, 1, jitter])
            ))
        }
        try linear.add(entries)
        try partitioned.add(entries)

        let query = VectorMath.normalized([0.98, 0.05, 0, 0]) // firmly in cluster A
        let exact = try linear.search(query: query, k: 1)
        let approx = try partitioned.search(query: query, k: 1)
        XCTAssertEqual(exact.first?.chunk.id, approx.first?.chunk.id)
    }

    func testPartitionedEmptySearchAndKClamping() throws {
        let index = PartitionedVectorIndex(dimension: 4, centroidCount: 4, probeCount: 2)
        XCTAssertTrue(try index.search(query: e0, k: 3).isEmpty)
        try index.add([Fixtures.entry(id: "a", index: 0, vector: e0)])
        XCTAssertTrue(try index.search(query: e0, k: 0).isEmpty)
        XCTAssertEqual(try index.search(query: e0, k: 10).count, 1)
    }

    func testPartitionedDuplicateIDsSkippedAcrossPartitions() throws {
        let index = PartitionedVectorIndex(dimension: 4, centroidCount: 2, probeCount: 2)
        let entry = Fixtures.entry(id: "a", index: 0, vector: e0)
        XCTAssertEqual(try index.add([entry]), 1)
        XCTAssertEqual(try index.add([entry]), 0)
        XCTAssertEqual(index.count, 1)
        XCTAssertEqual(index.entries.count, 1)
    }

    func testPartitionedDimensionMismatchThrows() {
        let index = PartitionedVectorIndex(dimension: 4)
        XCTAssertThrowsError(try index.search(query: [1], k: 1))
        XCTAssertThrowsError(try index.add([Fixtures.entry(id: "x", index: 0, vector: [1])]))
    }

    func testProbeCountClampedToCentroidCount() {
        let index = PartitionedVectorIndex(dimension: 4, centroidCount: 2, probeCount: 99)
        XCTAssertEqual(index.probeCount, 2)
        let floor = PartitionedVectorIndex(dimension: 4, centroidCount: 3, probeCount: 0)
        XCTAssertEqual(floor.probeCount, 1)
    }

    // MARK: - Snapshot persistence

    private func temporaryFileURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ragkit-test-\(UUID().uuidString).json")
    }

    func testSnapshotRoundTrip() throws {
        let url = temporaryFileURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let store = IndexSnapshotStore(fileURL: url)
        let entries = [
            Fixtures.entry(id: "a", index: 0, vector: e0),
            Fixtures.entry(id: "b", index: 1, vector: e1),
        ]
        try store.save(entries: entries, dimension: 4)
        let loaded = try store.load(expectedDimension: 4)
        XCTAssertEqual(loaded, entries)
    }

    func testCorruptedSnapshotThrowsSnapshotCorrupted() throws {
        let url = temporaryFileURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("not json at all {{{".utf8).write(to: url)

        let store = IndexSnapshotStore(fileURL: url)
        XCTAssertThrowsError(try store.load(expectedDimension: 4)) { error in
            guard case .snapshotCorrupted = error as? RAGError else {
                return XCTFail("expected snapshotCorrupted, got \(error)")
            }
        }
    }

    func testDimensionMismatchedSnapshotRejected() throws {
        let url = temporaryFileURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let store = IndexSnapshotStore(fileURL: url)
        try store.save(entries: [Fixtures.entry(id: "a", index: 0, vector: e0)], dimension: 4)
        XCTAssertThrowsError(try store.load(expectedDimension: 8)) { error in
            guard case .snapshotCorrupted = error as? RAGError else {
                return XCTFail("expected snapshotCorrupted, got \(error)")
            }
        }
    }

    func testMissingSnapshotFileThrowsNotCrashes() {
        let store = IndexSnapshotStore(fileURL: temporaryFileURL())
        XCTAssertFalse(store.exists)
        XCTAssertThrowsError(try store.load(expectedDimension: 4))
    }
}
