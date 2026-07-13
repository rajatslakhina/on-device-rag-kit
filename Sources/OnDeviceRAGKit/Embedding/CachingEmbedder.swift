import Foundation

/// LRU-caching, request-coalescing decorator around any `Embedder`.
///
/// Why this exists: on-device embedding is the most expensive step of the
/// pipeline (a real `NLContextualEmbedding`/MLX forward pass per chunk).
/// Re-embedding identical text — repeated queries, re-ingested documents —
/// is pure waste of battery and latency.
///
/// Two concurrency problems are handled explicitly:
///
/// 1. **Duplicate in-flight work.** Actor isolation alone does NOT prevent two
///    concurrent `embed` calls for the same text from both missing the cache:
///    the actor is re-entrant across the `await` into the wrapped embedder, so
///    a naive check-then-act would compute the same embedding twice. An
///    in-flight registry closes that gap — the second caller awaits the first
///    caller's batch task instead of starting its own.
///
///    **Failure semantics, stated precisely:** the first caller to observe a
///    batch failure synchronously evicts the *entire batch* (every sibling
///    text) from the registry before rethrowing, so a transient failure costs
///    at most one failed call per batch — never one per text — and the next
///    request for any of those texts re-embeds fresh. Standard single-flight
///    caveat: a caller that joins an in-flight computation receives that
///    computation's outcome, including its error, even if a fresh attempt at
///    join time might have succeeded.
///
/// 2. **Memory pressure.** `trim(toFraction:)` evicts the least-recently-used
///    portion of the cache without a full flush, so a memory warning degrades
///    hit rate instead of zeroing it. (Rejected alternative: `removeAll()` on
///    warning — simpler, but the very next query re-embeds everything at the
///    worst possible moment, when the system is already under pressure.)
public actor CachingEmbedder: Embedder {
    public let dimension: Int

    private let base: any Embedder
    private let capacity: Int

    private var cache: [String: [Float]] = [:]
    /// Recency order, least-recently-used first. O(n) touch is a deliberate
    /// simplicity trade — at realistic capacities (hundreds to a few thousand
    /// entries) a linked-list LRU buys nothing measurable here.
    private var recency: [String] = []

    /// One registration per text currently being embedded. Each entry knows
    /// its whole batch, so whoever first observes the batch's failure can
    /// evict every sibling deterministically (see failure semantics above).
    private struct InFlightEntry {
        let batchID: UUID
        let batchTexts: [String]
        let position: Int
        let task: Task<[[Float]], Error>
    }
    private var inFlight: [String: InFlightEntry] = [:]

    /// Observability counters — surfaced so the demo (and tests) can prove
    /// cache behavior instead of asserting it.
    public private(set) var hitCount = 0
    public private(set) var missCount = 0

    public init(wrapping base: any Embedder, capacity: Int = 512) {
        self.base = base
        self.dimension = base.dimension
        self.capacity = max(1, capacity)
    }

    public var count: Int { cache.count }

    public func embed(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        let uniqueOrder = orderedUnique(texts)

        // Kick off ONE batched task for all texts that are neither cached nor
        // already being computed. Batching matters: real embedders amortize
        // model-invocation overhead across a batch. No `await` happens between
        // the miss check and registration, so the check-then-register step is
        // atomic within the actor.
        let newMisses = uniqueOrder.filter { cache[$0] == nil && inFlight[$0] == nil }
        if !newMisses.isEmpty {
            let embedder = base
            let batch = newMisses
            let batchID = UUID()
            let task = Task { try await embedder.embed(batch) }
            for (position, text) in batch.enumerated() {
                inFlight[text] = InFlightEntry(
                    batchID: batchID,
                    batchTexts: batch,
                    position: position,
                    task: task
                )
            }
        }

        var resolved: [String: [Float]] = [:]
        for text in uniqueOrder {
            if let cached = cache[text] {
                hitCount += 1
                touch(text)
                resolved[text] = cached
                continue
            }
            missCount += 1
            guard let entry = inFlight[text] else {
                // A concurrent caller resolved this text between our
                // registration pass and now — it must be in the cache.
                if let cached = cache[text] {
                    resolved[text] = cached
                    continue
                }
                throw RAGError.invalidConfiguration("in-flight registry lost a pending text")
            }
            do {
                let vectors = try await entry.task.value
                guard entry.position >= 0, entry.position < vectors.count else {
                    throw RAGError.embeddingCountMismatch(
                        expected: entry.batchTexts.count, got: vectors.count
                    )
                }
                let vector = vectors[entry.position]
                guard vector.count == dimension else {
                    throw RAGError.dimensionMismatch(expected: dimension, got: vector.count)
                }
                store(text: text, vector: vector)
                resolved[text] = vector
                // Deregister only this text; siblings deregister as their own
                // awaiting callers resolve them (a completed-successfully task
                // left registered is harmless — it resolves instantly).
                if inFlight[text]?.batchID == entry.batchID {
                    inFlight[text] = nil
                }
            } catch {
                // First observer of a batch failure evicts the WHOLE batch —
                // synchronously, before rethrowing — so a dead task can never
                // poison siblings on later requests.
                evictBatch(entry)
                throw error
            }
        }

        // Rebuild the caller's order (duplicates included).
        var output: [[Float]] = []
        output.reserveCapacity(texts.count)
        for text in texts {
            guard let vector = resolved[text] else {
                throw RAGError.embeddingCountMismatch(expected: texts.count, got: output.count)
            }
            output.append(vector)
        }
        return output
    }

    /// Evicts least-recently-used entries until at most `fraction` of capacity
    /// remains. `trim(toFraction: 0)` empties the cache; values are clamped.
    public func trim(toFraction fraction: Double) {
        let clamped = min(max(fraction, 0), 1)
        let keep = Int((Double(capacity) * clamped).rounded(.down))
        while cache.count > keep, let oldest = recency.first {
            recency.removeFirst()
            cache[oldest] = nil
        }
    }

    // MARK: - Internals

    /// Removes every registration belonging to `entry`'s batch. The batchID
    /// check ensures a text that was already evicted and re-registered under
    /// a NEWER batch is left alone.
    private func evictBatch(_ entry: InFlightEntry) {
        for text in entry.batchTexts where inFlight[text]?.batchID == entry.batchID {
            inFlight[text] = nil
        }
    }

    private func store(text: String, vector: [Float]) {
        if cache[text] == nil, cache.count >= capacity, let oldest = recency.first {
            recency.removeFirst()
            cache[oldest] = nil
        }
        cache[text] = vector
        touch(text)
    }

    private func touch(_ text: String) {
        if let existing = recency.firstIndex(of: text) {
            recency.remove(at: existing)
        }
        recency.append(text)
    }

    private func orderedUnique(_ texts: [String]) -> [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for text in texts where !seen.contains(text) {
            seen.insert(text)
            ordered.append(text)
        }
        return ordered
    }
}
