import Foundation

/// Persists the index's entries to disk so a corpus survives relaunch without
/// re-chunking and re-embedding everything — on-device, embeddings are the
/// expensive artifact, so throwing them away between launches wastes battery.
///
/// Format: versioned JSON envelope. **Rejected alternative:** a memory-mapped
/// binary format — faster to load at large scale, but unauditable in a code
/// review, fragile across app versions, and unnecessary below tens of
/// thousands of vectors. JSON keeps the snapshot debuggable (`cat` it) and the
/// version field gives a forward-migration hook; the store is a protocol-free
/// final class because persistence policy, unlike embedding or indexing, has
/// exactly one sensible implementation at this scale.
public final class IndexSnapshotStore {
    /// Bumped if the on-disk layout ever changes; loads of a different version
    /// fail loudly with `snapshotCorrupted` instead of misdecoding silently.
    public static let currentVersion = 1

    struct Envelope: Codable {
        let version: Int
        let dimension: Int
        let entries: [EmbeddedChunk]
    }

    private let fileURL: URL

    public init(fileURL: URL) {
        self.fileURL = fileURL
    }

    /// Atomically writes all entries. Atomic matters: a crash mid-write must
    /// leave the previous snapshot intact, not a half-written file.
    public func save(entries: [EmbeddedChunk], dimension: Int) throws {
        let envelope = Envelope(
            version: Self.currentVersion,
            dimension: dimension,
            entries: entries
        )
        let data = try JSONEncoder().encode(envelope)
        try data.write(to: fileURL, options: .atomic)
    }

    /// Loads and validates a snapshot.
    /// - Throws: `RAGError.snapshotCorrupted` for undecodable data, a version
    ///   mismatch, or any entry whose vector does not match `expectedDimension`.
    public func load(expectedDimension: Int) throws -> [EmbeddedChunk] {
        let data: Data
        do {
            data = try Data(contentsOf: fileURL)
        } catch {
            throw RAGError.snapshotCorrupted("unreadable file: \(error.localizedDescription)")
        }
        let envelope: Envelope
        do {
            envelope = try JSONDecoder().decode(Envelope.self, from: data)
        } catch {
            throw RAGError.snapshotCorrupted("undecodable JSON: \(error.localizedDescription)")
        }
        guard envelope.version == Self.currentVersion else {
            throw RAGError.snapshotCorrupted(
                "version \(envelope.version) unsupported (expected \(Self.currentVersion))"
            )
        }
        guard envelope.dimension == expectedDimension else {
            throw RAGError.snapshotCorrupted(
                "snapshot dimension \(envelope.dimension) != index dimension \(expectedDimension)"
            )
        }
        for entry in envelope.entries where entry.vector.count != expectedDimension {
            throw RAGError.snapshotCorrupted(
                "entry \(entry.id) has \(entry.vector.count)-dim vector (expected \(expectedDimension))"
            )
        }
        return envelope.entries
    }

    public var exists: Bool {
        FileManager.default.fileExists(atPath: fileURL.path)
    }
}
