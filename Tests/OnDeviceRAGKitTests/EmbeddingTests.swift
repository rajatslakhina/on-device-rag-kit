import XCTest
@testable import OnDeviceRAGKit

final class EmbeddingTests: XCTestCase {

    // MARK: - DeterministicHashEmbedder

    func testDeterministicAcrossInstances() async throws {
        let a = try await DeterministicHashEmbedder(dimension: 64).embed(["swift actors"])
        let b = try await DeterministicHashEmbedder(dimension: 64).embed(["swift actors"])
        XCTAssertEqual(a, b)
    }

    func testEmptyBatchReturnsEmpty() async throws {
        let out = try await DeterministicHashEmbedder().embed([])
        XCTAssertTrue(out.isEmpty)
    }

    func testEmptyStringProducesZeroVectorAndSafeCosine() async throws {
        let embedder = DeterministicHashEmbedder(dimension: 32)
        let vectors = try await embedder.embed([""])
        XCTAssertEqual(vectors.count, 1)
        let zero = vectors[0]
        XCTAssertEqual(zero.count, 32)
        XCTAssertTrue(zero.allSatisfy { $0 == 0 })
        // Cosine against a zero vector must be 0, not NaN/crash.
        let other = try await embedder.embed(["hello"])
        XCTAssertEqual(VectorMath.cosineSimilarity(zero, other[0]), 0)
    }

    func testVectorsAreNormalized() async throws {
        let vectors = try await DeterministicHashEmbedder(dimension: 64).embed(["normalize me please"])
        let magnitude = vectors[0].reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(magnitude, 1.0, accuracy: 0.001)
    }

    func testLexicallySimilarTextsScoreHigherThanUnrelated() async throws {
        let embedder = DeterministicHashEmbedder(dimension: 128)
        let vectors = try await embedder.embed([
            "the cache evicts least recently used entries",
            "cache eviction uses least recently used order",
            "espresso milk steaming builds microfoam",
        ])
        let related = VectorMath.cosineSimilarity(vectors[0], vectors[1])
        let unrelated = VectorMath.cosineSimilarity(vectors[0], vectors[2])
        XCTAssertGreaterThan(related, unrelated)
    }

    func testDimensionFloorIsClamped() {
        XCTAssertEqual(DeterministicHashEmbedder(dimension: 2).dimension, 8)
    }

    // MARK: - CachingEmbedder

    func testCacheHitAvoidsSecondBaseCall() async throws {
        let base = CountingEmbedder()
        let caching = CachingEmbedder(wrapping: base)
        _ = try await caching.embed(["alpha"])
        _ = try await caching.embed(["alpha"])
        let calls = await base.callCount
        XCTAssertEqual(calls, 1)
        let hits = await caching.hitCount
        XCTAssertEqual(hits, 1)
    }

    func testBatchMixesHitsAndMissesInOneBaseCall() async throws {
        let base = CountingEmbedder()
        let caching = CachingEmbedder(wrapping: base)
        _ = try await caching.embed(["alpha"])
        _ = try await caching.embed(["alpha", "beta", "gamma"])
        let calls = await base.callCount
        let texts = await base.textCount
        XCTAssertEqual(calls, 2)         // one call per embed() invocation
        XCTAssertEqual(texts, 3)         // 1 (alpha) + 2 (beta, gamma) — alpha not re-sent
    }

    func testDuplicateTextsWithinOneBatch() async throws {
        let base = CountingEmbedder()
        let caching = CachingEmbedder(wrapping: base)
        let out = try await caching.embed(["same", "same", "same"])
        XCTAssertEqual(out.count, 3)
        XCTAssertEqual(out[0], out[1])
        XCTAssertEqual(out[1], out[2])
        let texts = await base.textCount
        XCTAssertEqual(texts, 1)
    }

    func testOutputOrderMatchesInputOrder() async throws {
        let base = DeterministicHashEmbedder(dimension: 64)
        let caching = CachingEmbedder(wrapping: base)
        let direct = try await base.embed(["one", "two", "three"])
        let cached = try await caching.embed(["one", "two", "three"])
        XCTAssertEqual(direct, cached)
        // And again with a permuted order, now served partly from cache.
        let permuted = try await caching.embed(["three", "one", "two"])
        XCTAssertEqual(permuted, [direct[2], direct[0], direct[1]])
    }

    func testConcurrentRequestsForSameTextCoalesceToOneBaseCall() async throws {
        let base = CountingEmbedder(delayNanoseconds: 50_000_000) // 50ms window
        let caching = CachingEmbedder(wrapping: base)
        async let first = caching.embed(["shared text"])
        async let second = caching.embed(["shared text"])
        let (a, b) = try await (first, second)
        XCTAssertEqual(a, b)
        let calls = await base.callCount
        XCTAssertEqual(calls, 1, "second caller should await the in-flight task, not re-embed")
    }

    func testFailureDoesNotPoisonFutureRequests() async throws {
        let base = FlakyEmbedder(failuresBeforeSuccess: 1)
        let caching = CachingEmbedder(wrapping: base)
        do {
            _ = try await caching.embed(["text"])
            XCTFail("expected first call to fail")
        } catch {
            // expected
        }
        let recovered = try await caching.embed(["text"])
        XCTAssertEqual(recovered.count, 1)
        let calls = await base.callCount
        XCTAssertEqual(calls, 2)
    }

    func testFailedBatchEvictsAllSiblingTextsNotJustTheFirst() async throws {
        // Regression test for a real bug the quality gate caught: a failed
        // multi-text batch used to evict only the first awaited text, leaving
        // every sibling registered against the dead task — poisoning each
        // sibling for exactly one extra request.
        let base = FlakyEmbedder(failuresBeforeSuccess: 1)
        let caching = CachingEmbedder(wrapping: base)
        do {
            _ = try await caching.embed(["sibling-1", "sibling-2", "sibling-3"])
            XCTFail("expected batch failure")
        } catch {
            XCTAssertTrue(error is TestError)
        }
        // EVERY sibling must recover on its very next request — one fresh
        // base call each, no stale rethrow from the dead batch task.
        let second = try await caching.embed(["sibling-2"])
        XCTAssertEqual(second.count, 1)
        let third = try await caching.embed(["sibling-3", "sibling-1"])
        XCTAssertEqual(third.count, 2)
        let calls = await base.callCount
        XCTAssertEqual(calls, 3, "1 failed batch + 2 fresh recovery calls")
    }

    func testCapacityEvictionIsLRU() async throws {
        let base = CountingEmbedder()
        let caching = CachingEmbedder(wrapping: base, capacity: 2)
        _ = try await caching.embed(["a"])
        _ = try await caching.embed(["b"])
        _ = try await caching.embed(["a"])      // touch a → b is now LRU
        _ = try await caching.embed(["c"])      // evicts b
        let callsBefore = await base.callCount
        _ = try await caching.embed(["a"])      // still cached — no new call
        let callsAfterA = await base.callCount
        XCTAssertEqual(callsAfterA, callsBefore)
        _ = try await caching.embed(["b"])      // evicted — requires a new call
        let callsAfterB = await base.callCount
        XCTAssertEqual(callsAfterB, callsBefore + 1)
    }

    func testTrimToFractionEvictsOldestFirst() async throws {
        let base = CountingEmbedder()
        let caching = CachingEmbedder(wrapping: base, capacity: 4)
        _ = try await caching.embed(["one", "two", "three", "four"])
        await caching.trim(toFraction: 0.5)     // keep 2 → "one","two" evicted
        let remaining = await caching.count
        XCTAssertEqual(remaining, 2)
        let callsBefore = await base.callCount
        _ = try await caching.embed(["four"])   // survivor — cache hit
        let callsAfter = await base.callCount
        XCTAssertEqual(callsAfter, callsBefore)
    }

    func testTrimToZeroEmptiesCache() async throws {
        let caching = CachingEmbedder(wrapping: CountingEmbedder(), capacity: 4)
        _ = try await caching.embed(["one", "two"])
        await caching.trim(toFraction: 0)
        let remaining = await caching.count
        XCTAssertEqual(remaining, 0)
    }
}
