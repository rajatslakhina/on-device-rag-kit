import Foundation

/// Produces dense vectors for texts.
///
/// This is the pipeline's most important seam: on Apple platforms the real
/// implementation is `NLContextualEmbedding` (NaturalLanguage) or an MLX
/// embedding model; a server-backed embedder is a third option for capable
/// networks. All three plug in here without the rest of the pipeline knowing.
///
/// Contract:
/// - Must return exactly one vector per input text, in input order.
/// - Every vector must have `dimension` elements.
/// - `embed([])` returns `[]` and must not throw.
public protocol Embedder: Sendable {
    var dimension: Int { get }
    func embed(_ texts: [String]) async throws -> [[Float]]
}

/// Deterministic, dependency-free embedder used as the default backend and in tests.
///
/// It hashes word unigrams and character trigrams into a fixed-size bucket
/// space (FNV-1a — deliberately NOT Swift's `hashValue`, which is randomly
/// seeded per process and would make embeddings non-reproducible across
/// launches, silently corrupting any persisted index) and L2-normalizes.
///
/// This is a bag-of-features embedding: it captures lexical overlap, not
/// semantics. That is exactly enough to exercise and test the *system* —
/// retrieval, ranking, budgeting, persistence — while remaining reproducible
/// on any platform. The README documents why this trade was chosen over
/// bundling a real model.
public struct DeterministicHashEmbedder: Embedder {
    public let dimension: Int

    public init(dimension: Int = 128) {
        self.dimension = max(8, dimension)
    }

    public func embed(_ texts: [String]) async throws -> [[Float]] {
        texts.map { vector(for: $0) }
    }

    // MARK: - Internals

    private func vector(for text: String) -> [Float] {
        var buckets = [Float](repeating: 0, count: dimension)
        let lowered = text.lowercased()
        let words = lowered
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }

        for word in words {
            buckets[Int(fnv1a(word) % UInt64(dimension))] += 1.0
            // Character trigrams give partial-match signal for morphology
            // ("index"/"indexing") that pure unigrams miss.
            let chars = Array(word)
            if chars.count >= 3 {
                for i in 0...(chars.count - 3) {
                    let gram = String(chars[i...(i + 2)])
                    buckets[Int(fnv1a("g:" + gram) % UInt64(dimension))] += 0.5
                }
            }
        }
        return VectorMath.normalized(buckets)
    }

    private func fnv1a(_ string: String) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in string.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x100000001b3
        }
        return hash
    }
}
