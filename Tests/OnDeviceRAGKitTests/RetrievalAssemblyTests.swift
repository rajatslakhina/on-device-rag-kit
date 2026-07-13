import XCTest
@testable import OnDeviceRAGKit

final class RetrievalAssemblyTests: XCTestCase {

    private func candidate(id: String, vector: [Float], score: Float, text: String = "text") -> RetrievedChunk {
        RetrievedChunk(
            chunk: Chunk(documentID: id, documentTitle: id, text: text, index: 0),
            vector: vector,
            score: score
        )
    }

    // MARK: - MMRReranker

    func testEmptyCandidatesReturnEmpty() {
        let reranker = MMRReranker(lambda: 0.7)
        XCTAssertTrue(reranker.rerank(query: [1, 0], candidates: [], limit: 5).isEmpty)
    }

    func testLimitZeroReturnsEmpty() {
        let reranker = MMRReranker(lambda: 0.7)
        let candidates = [candidate(id: "a", vector: [1, 0], score: 0.9)]
        XCTAssertTrue(reranker.rerank(query: [1, 0], candidates: candidates, limit: 0).isEmpty)
    }

    func testLambdaIsClamped() {
        XCTAssertEqual(MMRReranker(lambda: 7).lambda, 1)
        XCTAssertEqual(MMRReranker(lambda: -3).lambda, 0)
    }

    func testPureRelevanceLambdaPreservesSimilarityOrder() {
        let reranker = MMRReranker(lambda: 1.0)
        let candidates = [
            candidate(id: "best", vector: [1, 0, 0], score: 0.95),
            candidate(id: "mid", vector: [0.9, 0.1, 0], score: 0.80),
            candidate(id: "worst", vector: [0, 1, 0], score: 0.40),
        ]
        let out = reranker.rerank(query: [1, 0, 0], candidates: candidates, limit: 3)
        XCTAssertEqual(out.map { $0.chunk.documentID }, ["best", "mid", "worst"])
    }

    func testMMRPrefersDiverseSecondPickOverNearDuplicate() {
        // "dupe" is nearly identical to "best"; "different" is less relevant
        // but adds new information. With diversity-weighting on, MMR must pick
        // "different" second — a raw top-k would pick "dupe".
        let best = candidate(id: "best", vector: VectorMath.normalized([1, 0.01, 0]), score: 0.95)
        let dupe = candidate(id: "dupe", vector: VectorMath.normalized([1, 0.02, 0]), score: 0.94)
        let different = candidate(id: "different", vector: VectorMath.normalized([0, 0, 1]), score: 0.55)

        let reranker = MMRReranker(lambda: 0.5)
        let out = reranker.rerank(
            query: [1, 0, 0],
            candidates: [best, dupe, different],
            limit: 2
        )
        XCTAssertEqual(out.map { $0.chunk.documentID }, ["best", "different"])
    }

    func testRerankIsDeterministic() {
        let candidates = [
            candidate(id: "a", vector: [1, 0], score: 0.5),
            candidate(id: "b", vector: [0, 1], score: 0.5),
        ]
        let reranker = MMRReranker(lambda: 0.7)
        let first = reranker.rerank(query: [1, 0], candidates: candidates, limit: 2)
        let second = reranker.rerank(query: [1, 0], candidates: candidates, limit: 2)
        XCTAssertEqual(first.map { $0.id }, second.map { $0.id })
    }

    func testSimilarityRerankerTruncatesOnly() {
        let candidates = [
            candidate(id: "a", vector: [1, 0], score: 0.9),
            candidate(id: "b", vector: [0, 1], score: 0.8),
            candidate(id: "c", vector: [0, 1], score: 0.7),
        ]
        let out = SimilarityReranker().rerank(query: [1, 0], candidates: candidates, limit: 2)
        XCTAssertEqual(out.map { $0.chunk.documentID }, ["a", "b"])
    }

    // MARK: - CharacterHeuristicTokenizer

    func testTokenizerEmptyStringIsZero() {
        XCTAssertEqual(CharacterHeuristicTokenizer().estimateTokens(in: ""), 0)
    }

    func testTokenizerRoundsUp() {
        let tokenizer = CharacterHeuristicTokenizer(charactersPerToken: 4)
        XCTAssertEqual(tokenizer.estimateTokens(in: "abcde"), 2) // 5 chars → 2 tokens
        XCTAssertEqual(tokenizer.estimateTokens(in: "abcd"), 1)
    }

    func testTokenizerClampsCharactersPerToken() {
        XCTAssertEqual(CharacterHeuristicTokenizer(charactersPerToken: 0).charactersPerToken, 1)
    }

    // MARK: - ContextAssembler

    func testZeroBudgetIncludesNothing() {
        let assembler = ContextAssembler(tokenBudget: 0)
        let out = assembler.assemble(
            query: "q",
            candidates: [candidate(id: "a", vector: [1], score: 0.9, text: "hello world")]
        )
        XCTAssertTrue(out.includedChunks.isEmpty)
        XCTAssertEqual(out.droppedCount, 1)
        XCTAssertEqual(out.contextText, "")
    }

    func testEmptyCandidatesProduceEmptyContext() {
        let out = ContextAssembler(tokenBudget: 100).assemble(query: "q", candidates: [])
        XCTAssertTrue(out.includedChunks.isEmpty)
        XCTAssertEqual(out.droppedCount, 0)
        XCTAssertEqual(out.estimatedTokens, 0)
    }

    func testGreedySkipStillAdmitsSmallerLaterChunks() {
        // Budget fits chunk A (20 chars ≈ 5 tokens) and chunk C (8 chars ≈ 2
        // tokens) but not chunk B (200 chars ≈ 50 tokens). Greedy-skip must
        // include A and C; greedy-stop would have quit at B.
        let assembler = ContextAssembler(tokenBudget: 8)
        let a = candidate(id: "a", vector: [1], score: 0.9, text: String(repeating: "x", count: 20))
        let b = candidate(id: "b", vector: [1], score: 0.8, text: String(repeating: "y", count: 200))
        let c = candidate(id: "c", vector: [1], score: 0.7, text: String(repeating: "z", count: 8))
        let out = assembler.assemble(query: "q", candidates: [a, b, c])
        XCTAssertEqual(out.includedChunks.map { $0.chunk.documentID }, ["a", "c"])
        XCTAssertEqual(out.droppedCount, 1)
    }

    func testOversizedFirstChunkIsTruncatedNotDropped() {
        let assembler = ContextAssembler(tokenBudget: 5) // 5 tokens ≈ 20 chars
        let huge = candidate(id: "a", vector: [1], score: 0.9, text: String(repeating: "w", count: 400))
        let out = assembler.assemble(query: "q", candidates: [huge])
        XCTAssertEqual(out.includedChunks.count, 1)
        XCTAssertLessThanOrEqual(out.includedChunks[0].chunk.text.count, 20)
        XCTAssertFalse(out.contextText.isEmpty)
    }

    func testDuplicateChunkIDsAreNotDoubleSpent() {
        let assembler = ContextAssembler(tokenBudget: 100)
        let a = candidate(id: "a", vector: [1], score: 0.9, text: "hello")
        let out = assembler.assemble(query: "q", candidates: [a, a, a])
        XCTAssertEqual(out.includedChunks.count, 1)
        XCTAssertEqual(out.droppedCount, 2)
    }

    func testContextTextCarriesDocumentTitles() {
        let assembler = ContextAssembler(tokenBudget: 100)
        let a = candidate(id: "guide", vector: [1], score: 0.9, text: "actors serialize access")
        let out = assembler.assemble(query: "q", candidates: [a])
        XCTAssertTrue(out.contextText.contains("[guide]"))
        XCTAssertTrue(out.contextText.contains("actors serialize access"))
    }
}
