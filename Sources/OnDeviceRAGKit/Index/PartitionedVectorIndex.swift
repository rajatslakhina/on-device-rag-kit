import Foundation

/// IVF-style approximate index: entries are bucketed under coarse centroids;
/// a query scans only the `probeCount` nearest partitions instead of the
/// whole corpus.
///
/// **The trade-off, stated plainly:** query cost drops from O(n·d) to roughly
/// O((c + n/c · p)·d) for c centroids probing p partitions — but recall is no
/// longer perfect. A true nearest neighbor that landed in an unprobed
/// partition is invisible to the query. That is the entire deal with ANN
/// search, and it is why `LinearVectorIndex` remains the default: on-device
/// corpora usually sit below the scale where this trade pays for itself.
///
/// **Rejected alternatives:**
/// - *HNSW* — better recall/latency frontier at large scale, but the graph
///   costs significant memory per entry and the implementation complexity is
///   not defensible for the corpus sizes an iOS app realistically indexes.
/// - *Offline k-means training* — higher-quality centroids, but requires a
///   training pass over a static corpus. This index must accept streaming
///   ingestion, so it uses online centroid updates (running means) instead:
///   the first `centroidCount` entries seed the centroids, every later entry
///   nudges its nearest centroid toward itself.
public final class PartitionedVectorIndex: VectorIndex {
    public let dimension: Int
    public let centroidCount: Int
    public let probeCount: Int

    private var centroids: [[Float]] = []
    /// Running count of members per centroid, for the online mean update.
    private var memberCounts: [Int] = []
    private var partitions: [[EmbeddedChunk]] = []
    private var knownIDs = Set<String>()

    public init(dimension: Int, centroidCount: Int = 8, probeCount: Int = 2) {
        self.dimension = max(1, dimension)
        self.centroidCount = max(1, centroidCount)
        self.probeCount = min(max(1, probeCount), max(1, centroidCount))
    }

    public var count: Int { knownIDs.count }
    public var entries: [EmbeddedChunk] { partitions.flatMap { $0 } }

    @discardableResult
    public func add(_ items: [EmbeddedChunk]) throws -> Int {
        // Validate the WHOLE batch before mutating anything (same atomic-add
        // contract as LinearVectorIndex): a mixed valid/invalid batch must
        // never leave a partially-inserted document behind.
        for item in items {
            guard item.vector.count == dimension else {
                throw RAGError.dimensionMismatch(expected: dimension, got: item.vector.count)
            }
        }
        var inserted = 0
        for item in items {
            guard !knownIDs.contains(item.id) else { continue }
            knownIDs.insert(item.id)

            if centroids.count < centroidCount {
                // Seed a new partition with this vector as its centroid.
                centroids.append(item.vector)
                memberCounts.append(1)
                partitions.append([item])
            } else {
                let target = nearestCentroids(to: item.vector, count: 1)
                // `nearestCentroids` returns at least one index when centroids
                // is non-empty, which is guaranteed on this branch — but
                // bounds-check anyway instead of force-unwrapping.
                guard let partitionIndex = target.first,
                      partitionIndex >= 0, partitionIndex < partitions.count else {
                    throw RAGError.invalidConfiguration("partition routing failed")
                }
                partitions[partitionIndex].append(item)
                updateCentroid(at: partitionIndex, with: item.vector)
            }
            inserted += 1
        }
        return inserted
    }

    public func search(query: [Float], k: Int) throws -> [RetrievedChunk] {
        guard query.count == dimension else {
            throw RAGError.dimensionMismatch(expected: dimension, got: query.count)
        }
        guard k > 0, !knownIDs.isEmpty else { return [] }

        let probes = nearestCentroids(to: query, count: probeCount)
        var scored: [RetrievedChunk] = []
        for partitionIndex in probes where partitionIndex >= 0 && partitionIndex < partitions.count {
            for entry in partitions[partitionIndex] {
                scored.append(
                    RetrievedChunk(
                        chunk: entry.chunk,
                        vector: entry.vector,
                        score: VectorMath.cosineSimilarity(query, entry.vector)
                    )
                )
            }
        }
        let sorted = scored.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.chunk.id < $1.chunk.id
        }
        return Array(sorted.prefix(k))
    }

    public func removeAll() {
        centroids.removeAll()
        memberCounts.removeAll()
        partitions.removeAll()
        knownIDs.removeAll()
    }

    // MARK: - Internals

    /// Indices of the `count` centroids nearest to `vector`, best first.
    private func nearestCentroids(to vector: [Float], count: Int) -> [Int] {
        guard !centroids.isEmpty else { return [] }
        let ranked = centroids.enumerated()
            .map { (index: $0.offset, score: VectorMath.cosineSimilarity(vector, $0.element)) }
            .sorted {
                if $0.score != $1.score { return $0.score > $1.score }
                return $0.index < $1.index
            }
        return ranked.prefix(max(1, count)).map { $0.index }
    }

    /// Online running-mean update: centroid += (vector - centroid) / n.
    private func updateCentroid(at index: Int, with vector: [Float]) {
        guard index >= 0, index < centroids.count, index < memberCounts.count else { return }
        memberCounts[index] += 1
        let n = Float(memberCounts[index])
        guard centroids[index].count == vector.count else { return }
        for d in 0..<vector.count {
            centroids[index][d] += (vector[d] - centroids[index][d]) / n
        }
    }
}
