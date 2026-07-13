import Foundation

/// Shared vector math. Kept dependency-free (no Accelerate/vDSP) so every
/// component builds and tests on Linux CI; the functions are isolated here so
/// an app target can swap in a vDSP-backed implementation behind the same
/// call sites if profiling ever shows the scalar loops matter.
public enum VectorMath {
    /// Cosine similarity in [-1, 1]. Degenerate inputs (zero-magnitude vectors,
    /// mismatched lengths) return 0 rather than trapping or dividing by zero —
    /// a zero vector is a legitimate output for empty text.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var magA: Float = 0
        var magB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            magA += a[i] * a[i]
            magB += b[i] * b[i]
        }
        let denominator = magA.squareRoot() * magB.squareRoot()
        guard denominator > 0 else { return 0 }
        return dot / denominator
    }

    /// L2-normalizes; a zero vector normalizes to itself (no division by zero).
    public static func normalized(_ vector: [Float]) -> [Float] {
        var magnitude: Float = 0
        for value in vector {
            magnitude += value * value
        }
        let root = magnitude.squareRoot()
        guard root > 0 else { return vector }
        return vector.map { $0 / root }
    }
}

/// Stores embedded chunks and answers nearest-neighbor queries.
///
/// Implementations are exclusively owned by the `RAGPipeline` actor once
/// handed to it — the pipeline's actor isolation is the single concurrency
/// boundary for index state. (Rejected alternative: making each index its own
/// actor — that double-hops every query through two actors and reintroduces
/// ordering questions between the hops for no isolation benefit, since the
/// index is never shared outside the pipeline.)
public protocol VectorIndex: AnyObject {
    var dimension: Int { get }
    var count: Int { get }
    /// Every stored entry — used for snapshot persistence.
    var entries: [EmbeddedChunk] { get }

    /// Adds entries, skipping any whose chunk ID is already present
    /// (re-ingesting a document is idempotent, not duplicating).
    /// Returns the number actually inserted.
    /// - Throws: `RAGError.dimensionMismatch` if any vector has the wrong size.
    @discardableResult
    func add(_ items: [EmbeddedChunk]) throws -> Int

    /// Returns up to `k` nearest entries by cosine similarity, best first.
    /// `k <= 0` or an empty index returns `[]`; `k > count` clamps.
    /// - Throws: `RAGError.dimensionMismatch` for a wrong-sized query vector.
    func search(query: [Float], k: Int) throws -> [RetrievedChunk]

    func removeAll()
}

/// Exact brute-force index: O(n·d) per query, perfect recall.
///
/// This is the right default for on-device corpora. At d=128, a linear scan
/// over 50k chunks is a few million multiply-adds — well under a frame budget
/// on any A-series chip. ANN structures only earn their complexity above that
/// scale (see `PartitionedVectorIndex` for the trade-off, and the README for
/// why HNSW was rejected outright).
public final class LinearVectorIndex: VectorIndex {
    public let dimension: Int
    private var storage: [EmbeddedChunk] = []
    private var knownIDs = Set<String>()

    public init(dimension: Int) {
        self.dimension = max(1, dimension)
    }

    public var count: Int { storage.count }
    public var entries: [EmbeddedChunk] { storage }

    @discardableResult
    public func add(_ items: [EmbeddedChunk]) throws -> Int {
        // Validate the WHOLE batch before mutating anything, so a batch that
        // mixes valid and invalid dimensions can never leave a partial insert
        // behind — `add` is atomic, which is what lets RAGPipeline promise
        // per-document ingest atomicity without re-validating here.
        for item in items {
            guard item.vector.count == dimension else {
                throw RAGError.dimensionMismatch(expected: dimension, got: item.vector.count)
            }
        }
        var inserted = 0
        for item in items {
            guard !knownIDs.contains(item.id) else { continue }
            knownIDs.insert(item.id)
            storage.append(item)
            inserted += 1
        }
        return inserted
    }

    public func search(query: [Float], k: Int) throws -> [RetrievedChunk] {
        guard query.count == dimension else {
            throw RAGError.dimensionMismatch(expected: dimension, got: query.count)
        }
        guard k > 0, !storage.isEmpty else { return [] }

        let scored = storage.map { entry in
            RetrievedChunk(
                chunk: entry.chunk,
                vector: entry.vector,
                score: VectorMath.cosineSimilarity(query, entry.vector)
            )
        }
        // Deterministic ordering: score descending, chunk ID as tie-break so
        // identical corpora always retrieve identically (matters for tests,
        // snapshots, and reproducing user-reported retrieval bugs).
        let sorted = scored.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.chunk.id < $1.chunk.id
        }
        return Array(sorted.prefix(k))
    }

    public func removeAll() {
        storage.removeAll()
        knownIDs.removeAll()
    }
}
