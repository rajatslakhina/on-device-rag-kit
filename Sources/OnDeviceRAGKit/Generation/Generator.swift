import Foundation

/// Produces a streamed answer from an assembled context.
///
/// On Apple platforms the real implementation wraps a
/// `LanguageModelSession` (Foundation Models) or an MLX-hosted model; the
/// stream shape matches what those APIs emit so swapping the backend does not
/// ripple into the UI. The pipeline treats generation as opaque: everything
/// retrieval-related happens before this seam.
public protocol Generator: Sendable {
    func generate(context: AssembledContext) -> AsyncThrowingStream<String, Error>
}

/// Deterministic template generator used as the default backend and in tests.
///
/// It streams a grounded answer that cites the retrieved passages — enough to
/// demonstrate (and test) the full retrieval → assembly → streaming path
/// without bundling a language model. The README documents this boundary
/// honestly: this library's product is the *retrieval system*; generation
/// quality belongs to whichever model the app plugs in.
public struct SimulatedGenerator: Generator {
    /// Delay between streamed tokens. Zero in tests; the demo app uses a small
    /// delay so streaming is visible in the UI.
    public let tokenDelayNanoseconds: UInt64

    public init(tokenDelayNanoseconds: UInt64 = 0) {
        self.tokenDelayNanoseconds = tokenDelayNanoseconds
    }

    public func generate(context: AssembledContext) -> AsyncThrowingStream<String, Error> {
        let text = composeAnswer(for: context)
        let delay = tokenDelayNanoseconds
        return AsyncThrowingStream { continuation in
            let task = Task {
                // Stream word-by-word to exercise real incremental-UI paths.
                let tokens = text.split(separator: " ", omittingEmptySubsequences: false)
                for (index, token) in tokens.enumerated() {
                    if Task.isCancelled {
                        continuation.finish(throwing: CancellationError())
                        return
                    }
                    if delay > 0 {
                        try await Task.sleep(nanoseconds: delay)
                    }
                    continuation.yield(index == 0 ? String(token) : " " + String(token))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    // MARK: - Internals

    private func composeAnswer(for context: AssembledContext) -> String {
        guard !context.includedChunks.isEmpty else {
            return "I could not find any indexed passage relevant to \u{201C}\(context.query)\u{201D}. "
                + "Try ingesting a document that covers this topic, or rephrasing the query."
        }
        let titles = orderedUniqueTitles(in: context.includedChunks)
        let sourceList = titles.joined(separator: ", ")
        let top = context.includedChunks[0] // non-empty: guarded above
        let excerpt = String(top.chunk.text.prefix(160))
        return "Based on \(context.includedChunks.count) retrieved passage(s) from \(sourceList): "
            + "\(excerpt)\u{2026} "
            + "(top passage \(top.chunk.id), similarity \(String(format: "%.3f", top.score)); "
            + "~\(context.estimatedTokens) context tokens used, \(context.droppedCount) candidate(s) dropped by budget)"
    }

    private func orderedUniqueTitles(in chunks: [RetrievedChunk]) -> [String] {
        var seen = Set<String>()
        var titles: [String] = []
        for retrieved in chunks {
            let title = retrieved.chunk.documentTitle
            if !seen.contains(title) {
                seen.insert(title)
                titles.append(title)
            }
        }
        return titles
    }
}
