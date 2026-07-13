import Foundation

/// Estimates token counts for budget enforcement.
public protocol TokenEstimating: Sendable {
    func estimateTokens(in text: String) -> Int
}

/// Characters-per-token heuristic (default 4, the common English estimate).
///
/// **Rejected alternative:** shipping a real tokenizer vocabulary — exact, but
/// couples this library to one model family's vocabulary and adds a bundled
/// asset. The heuristic's error bars (roughly ±20% on English prose) are
/// absorbed by treating the budget as a soft ceiling set below the model's
/// hard context limit. The protocol seam is where a real tokenizer plugs in
/// when the generator is known.
public struct CharacterHeuristicTokenizer: TokenEstimating {
    public let charactersPerToken: Int

    public init(charactersPerToken: Int = 4) {
        self.charactersPerToken = max(1, charactersPerToken)
    }

    public func estimateTokens(in text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        // Round up: underestimating token usage risks blowing the real
        // context limit, which fails worse than wasting a little budget.
        return (text.count + charactersPerToken - 1) / charactersPerToken
    }
}

/// Packs reranked chunks into a context block under a token budget.
///
/// Policy decisions, each deliberate:
/// - **Greedy-skip, not greedy-stop:** a chunk that doesn't fit is skipped and
///   later (smaller) chunks still get a chance. Stopping at the first miss
///   wastes remaining budget for no reason.
/// - **Oversized-first-chunk truncation:** if the single best chunk alone
///   exceeds the whole budget, it is truncated to fit rather than dropped —
///   an empty context when relevant material exists is the worst outcome.
/// - **Dedupe by chunk ID:** upstream bugs must not double-spend budget.
/// - **Budget <= 0 yields an empty context** rather than trapping.
public struct ContextAssembler: Sendable {
    public let tokenBudget: Int
    public let tokenizer: any TokenEstimating

    public init(tokenBudget: Int = 1024, tokenizer: any TokenEstimating = CharacterHeuristicTokenizer()) {
        self.tokenBudget = tokenBudget
        self.tokenizer = tokenizer
    }

    public func assemble(query: String, candidates: [RetrievedChunk]) -> AssembledContext {
        guard tokenBudget > 0, !candidates.isEmpty else {
            return AssembledContext(
                query: query,
                includedChunks: [],
                droppedCount: candidates.count,
                estimatedTokens: 0,
                contextText: ""
            )
        }

        var included: [RetrievedChunk] = []
        var seenIDs = Set<String>()
        var usedTokens = 0
        var dropped = 0

        for candidate in candidates {
            guard !seenIDs.contains(candidate.chunk.id) else {
                dropped += 1
                continue
            }
            let cost = tokenizer.estimateTokens(in: candidate.chunk.text)
            if usedTokens + cost <= tokenBudget {
                included.append(candidate)
                seenIDs.insert(candidate.chunk.id)
                usedTokens += cost
            } else if included.isEmpty {
                // Best chunk alone exceeds the budget: truncate instead of
                // returning an empty context.
                let truncated = truncate(candidate.chunk.text, toTokens: tokenBudget)
                let truncatedChunk = Chunk(
                    documentID: candidate.chunk.documentID,
                    documentTitle: candidate.chunk.documentTitle,
                    text: truncated,
                    index: candidate.chunk.index
                )
                included.append(
                    RetrievedChunk(
                        chunk: truncatedChunk,
                        vector: candidate.vector,
                        score: candidate.score
                    )
                )
                seenIDs.insert(candidate.chunk.id)
                usedTokens += tokenizer.estimateTokens(in: truncated)
            } else {
                dropped += 1
            }
        }

        let contextText = included
            .map { "[\($0.chunk.documentTitle)] \($0.chunk.text)" }
            .joined(separator: "\n\n---\n\n")

        return AssembledContext(
            query: query,
            includedChunks: included,
            droppedCount: dropped,
            estimatedTokens: usedTokens,
            contextText: contextText
        )
    }

    // MARK: - Internals

    /// Tokenizer-agnostic truncation: proportionally shrinks the text until
    /// its estimate fits the budget. Each pass strictly reduces the character
    /// count (the `count - 1` floor), so the loop provably terminates even if
    /// a custom tokenizer's estimates are non-linear.
    private func truncate(_ text: String, toTokens budget: Int) -> String {
        guard budget > 0 else { return "" }
        var candidate = text
        while !candidate.isEmpty, tokenizer.estimateTokens(in: candidate) > budget {
            let estimated = max(1, tokenizer.estimateTokens(in: candidate))
            let target = max(0, min(candidate.count - 1, candidate.count * budget / estimated))
            let end = candidate.index(
                candidate.startIndex, offsetBy: target, limitedBy: candidate.endIndex
            ) ?? candidate.endIndex
            candidate = String(candidate[candidate.startIndex..<end])
        }
        return candidate
    }
}
