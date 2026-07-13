import Foundation

/// Re-orders (and typically prunes) retrieved candidates before context
/// assembly. A separate seam from the index because ranking policy changes
/// far more often than storage layout — and because a cross-encoder reranker
/// (the quality ceiling) would plug in here when a model is available.
public protocol Reranker: Sendable {
    func rerank(query: [Float], candidates: [RetrievedChunk], limit: Int) -> [RetrievedChunk]
}

/// Pass-through reranker: keeps pure similarity order, just truncates.
public struct SimilarityReranker: Reranker {
    public init() {}

    public func rerank(query: [Float], candidates: [RetrievedChunk], limit: Int) -> [RetrievedChunk] {
        guard limit > 0 else { return [] }
        return Array(candidates.prefix(limit))
    }
}

/// Maximal Marginal Relevance: each pick balances relevance to the query
/// against redundancy with already-picked chunks.
///
///     MMR(c) = λ·sim(c, query) − (1−λ)·max sim(c, picked)
///
/// **Why it's here:** overlapping chunkers (see `FixedSizeChunker`) guarantee
/// that the top-k by raw similarity is often the *same passage k times*.
/// Feeding a generator five near-duplicates wastes most of the context budget
/// on one fact. MMR trades a little relevance for coverage.
///
/// **Rejected alternative:** cross-encoder reranking — strictly better
/// quality, but requires a second model forward-pass per candidate, which is
/// exactly the latency/battery cost this on-device pipeline is budgeting
/// against. MMR is O(k²) vector math on vectors already in hand.
public struct MMRReranker: Reranker {
    /// 1.0 = pure relevance (behaves like `SimilarityReranker`), 0.0 = pure
    /// diversity. Clamped to [0, 1] at init.
    public let lambda: Float

    public init(lambda: Float = 0.7) {
        self.lambda = min(max(lambda, 0), 1)
    }

    public func rerank(query: [Float], candidates: [RetrievedChunk], limit: Int) -> [RetrievedChunk] {
        guard limit > 0, !candidates.isEmpty else { return [] }

        var remaining = candidates
        var picked: [RetrievedChunk] = []
        picked.reserveCapacity(min(limit, candidates.count))

        while picked.count < limit, !remaining.isEmpty {
            var bestIndex = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for (index, candidate) in remaining.enumerated() {
                let relevance = candidate.score
                var redundancy: Float = 0
                for chosen in picked {
                    redundancy = max(
                        redundancy,
                        VectorMath.cosineSimilarity(candidate.vector, chosen.vector)
                    )
                }
                let mmr = lambda * relevance - (1 - lambda) * redundancy
                // Deterministic tie-break on chunk ID keeps reranking
                // reproducible across runs.
                if mmr > bestScore ||
                    (mmr == bestScore && candidate.chunk.id < remaining[bestIndex].chunk.id) {
                    bestScore = mmr
                    bestIndex = index
                }
            }
            // bestIndex is always a valid index of `remaining` here: the loop
            // requires `remaining` non-empty and bestIndex starts at 0.
            picked.append(remaining.remove(at: bestIndex))
        }
        return picked
    }
}
