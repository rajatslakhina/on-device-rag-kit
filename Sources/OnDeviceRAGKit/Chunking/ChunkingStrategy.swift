import Foundation

/// Splits a document into retrieval-sized chunks.
///
/// Chunking is a protocol seam because the right strategy is content-dependent
/// (prose vs. code vs. transcripts) and teams will want to swap it without
/// touching the rest of the pipeline.
public protocol ChunkingStrategy: Sendable {
    func chunk(_ document: RAGDocument) -> [Chunk]
}

/// Character-window chunker with overlap.
///
/// Overlap exists so that a sentence straddling a chunk boundary is fully
/// contained in at least one chunk — without it, boundary-straddling facts
/// become unretrievable. The cost is index size (each overlapped region is
/// embedded and stored twice), which is the documented trade-off.
///
/// Degenerate configurations are clamped rather than trapped:
/// - `chunkSize <= 0` clamps to 1
/// - `overlap` clamps into `[0, chunkSize - 1]` (an overlap >= chunkSize would
///   never advance the window and loop forever)
public struct FixedSizeChunker: ChunkingStrategy {
    public let chunkSize: Int
    public let overlap: Int

    public init(chunkSize: Int = 480, overlap: Int = 80) {
        let size = max(1, chunkSize)
        self.chunkSize = size
        self.overlap = min(max(0, overlap), size - 1)
    }

    public func chunk(_ document: RAGDocument) -> [Chunk] {
        let text = document.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }

        var chunks: [Chunk] = []
        var start = text.startIndex
        var chunkIndex = 0

        while start < text.endIndex {
            let end = text.index(start, offsetBy: chunkSize, limitedBy: text.endIndex) ?? text.endIndex
            let piece = String(text[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty {
                chunks.append(
                    Chunk(
                        documentID: document.id,
                        documentTitle: document.title,
                        text: piece,
                        index: chunkIndex
                    )
                )
                chunkIndex += 1
            }
            if end == text.endIndex { break }
            // Step forward by (chunkSize - overlap); clamping in init guarantees
            // this stride is >= 1, so the loop always terminates.
            start = text.index(start, offsetBy: chunkSize - overlap, limitedBy: text.endIndex) ?? text.endIndex
        }
        return chunks
    }
}

/// Paragraph-aware chunker: keeps paragraphs intact when they fit, merges small
/// neighbors up to `targetSize`, and delegates oversized paragraphs to a
/// `FixedSizeChunker` so no single chunk blows past the budget.
///
/// Rejected alternative: sentence-level splitting via `NLTokenizer` — it is
/// higher quality for prose but platform-locks the chunker to Apple OSes, and
/// this library keeps every component testable on Linux CI. The protocol seam
/// is exactly where an app target would swap in an `NLTokenizer`-based version.
public struct ParagraphChunker: ChunkingStrategy {
    public let targetSize: Int
    private let overflow: FixedSizeChunker

    public init(targetSize: Int = 480) {
        self.targetSize = max(1, targetSize)
        self.overflow = FixedSizeChunker(chunkSize: max(1, targetSize), overlap: max(0, targetSize / 6))
    }

    public func chunk(_ document: RAGDocument) -> [Chunk] {
        let paragraphs = document.text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !paragraphs.isEmpty else { return [] }

        var pieces: [String] = []
        var buffer = ""

        for paragraph in paragraphs {
            if paragraph.count > targetSize {
                // Flush whatever is buffered, then split the oversized paragraph.
                if !buffer.isEmpty {
                    pieces.append(buffer)
                    buffer = ""
                }
                let sub = overflow.chunk(
                    RAGDocument(id: document.id, title: document.title, text: paragraph)
                )
                pieces.append(contentsOf: sub.map { $0.text })
            } else if buffer.isEmpty {
                buffer = paragraph
            } else if buffer.count + 2 + paragraph.count <= targetSize {
                buffer += "\n\n" + paragraph
            } else {
                pieces.append(buffer)
                buffer = paragraph
            }
        }
        if !buffer.isEmpty {
            pieces.append(buffer)
        }

        return pieces.enumerated().map { offset, text in
            Chunk(
                documentID: document.id,
                documentTitle: document.title,
                text: text,
                index: offset
            )
        }
    }
}
