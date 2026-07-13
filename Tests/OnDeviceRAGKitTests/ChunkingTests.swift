import XCTest
@testable import OnDeviceRAGKit

final class ChunkingTests: XCTestCase {

    // MARK: - FixedSizeChunker

    func testEmptyDocumentProducesNoChunks() {
        let chunker = FixedSizeChunker(chunkSize: 100, overlap: 20)
        let doc = RAGDocument(id: "d", title: "t", text: "")
        XCTAssertTrue(chunker.chunk(doc).isEmpty)
    }

    func testWhitespaceOnlyDocumentProducesNoChunks() {
        let chunker = FixedSizeChunker(chunkSize: 100, overlap: 20)
        let doc = RAGDocument(id: "d", title: "t", text: "   \n\n\t  \n ")
        XCTAssertTrue(chunker.chunk(doc).isEmpty)
    }

    func testOverlapGreaterThanChunkSizeIsClampedAndTerminates() {
        // Without clamping this configuration would never advance the window.
        let chunker = FixedSizeChunker(chunkSize: 10, overlap: 50)
        XCTAssertEqual(chunker.overlap, 9)
        let doc = RAGDocument(id: "d", title: "t", text: String(repeating: "abcde ", count: 40))
        let chunks = chunker.chunk(doc)
        XCTAssertFalse(chunks.isEmpty)
    }

    func testZeroChunkSizeIsClamped() {
        let chunker = FixedSizeChunker(chunkSize: 0, overlap: 0)
        XCTAssertEqual(chunker.chunkSize, 1)
        let doc = RAGDocument(id: "d", title: "t", text: "ab")
        XCTAssertEqual(chunker.chunk(doc).count, 2)
    }

    func testTextExactlyOneChunkLong() {
        let chunker = FixedSizeChunker(chunkSize: 5, overlap: 0)
        let doc = RAGDocument(id: "d", title: "t", text: "abcde")
        let chunks = chunker.chunk(doc)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "abcde")
    }

    func testOverlappingChunksShareBoundaryContent() {
        let chunker = FixedSizeChunker(chunkSize: 10, overlap: 4)
        let doc = RAGDocument(id: "d", title: "t", text: "abcdefghijklmnopqrst")
        let chunks = chunker.chunk(doc)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        guard chunks.count >= 2 else { return }
        let firstTail = String(chunks[0].text.suffix(4))
        XCTAssertTrue(chunks[1].text.hasPrefix(firstTail))
    }

    func testChunkIDsAreDeterministicAcrossRuns() {
        let chunker = FixedSizeChunker(chunkSize: 8, overlap: 0)
        let doc = RAGDocument(id: "doc-1", title: "t", text: "abcdefghijklmnop")
        let first = chunker.chunk(doc).map { $0.id }
        let second = chunker.chunk(doc).map { $0.id }
        XCTAssertEqual(first, second)
        XCTAssertEqual(first.first, "doc-1#0")
    }

    // MARK: - ParagraphChunker

    func testParagraphChunkerEmptyDocument() {
        let chunker = ParagraphChunker(targetSize: 100)
        let doc = RAGDocument(id: "d", title: "t", text: "\n\n \n\n")
        XCTAssertTrue(chunker.chunk(doc).isEmpty)
    }

    func testSmallParagraphsAreMerged() {
        let chunker = ParagraphChunker(targetSize: 100)
        let doc = RAGDocument(id: "d", title: "t", text: "one two.\n\nthree four.\n\nfive six.")
        let chunks = chunker.chunk(doc)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertTrue(chunks.first?.text.contains("three four.") ?? false)
    }

    func testOversizedParagraphIsSplitNotDropped() {
        let chunker = ParagraphChunker(targetSize: 50)
        let big = String(repeating: "word ", count: 40) // ~200 chars, single paragraph
        let doc = RAGDocument(id: "d", title: "t", text: big)
        let chunks = chunker.chunk(doc)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.text.count, 50)
        }
    }

    func testParagraphBoundariesRespectedWhenTheyFit() {
        let chunker = ParagraphChunker(targetSize: 30)
        let doc = RAGDocument(
            id: "d", title: "t",
            text: "first paragraph here padding.\n\nsecond paragraph here padding."
        )
        let chunks = chunker.chunk(doc)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks.map { $0.index }, [0, 1])
    }
}
