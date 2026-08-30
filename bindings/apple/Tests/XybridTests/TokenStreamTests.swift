import Foundation
import XCTest
@testable import Xybrid

final class TokenStreamTests: XCTestCase {
    func testPullsExactlyOnceForEachRequestedElement() async throws {
        let probe = StreamProbe(events: [
            .token("one", index: 0),
            .token("two", index: 1),
            .complete,
        ])
        let stream = probe.stream()
        var iterator = stream.makeAsyncIterator()

        XCTAssertEqual(probe.startCount, 0)
        XCTAssertEqual(probe.pullCount, 0)

        let first = try await iterator.next()
        XCTAssertEqual(first?.token, "one")
        XCTAssertEqual(probe.startCount, 1)
        XCTAssertEqual(probe.pullCount, 1)

        await Task.yield()
        XCTAssertEqual(probe.pullCount, 1, "the stream must not prefetch")

        let second = try await iterator.next()
        XCTAssertEqual(second?.token, "two")
        XCTAssertEqual(probe.pullCount, 2)
        let end = try await iterator.next()
        XCTAssertNil(end)
        XCTAssertEqual(probe.pullCount, 3)
        XCTAssertEqual(probe.closeCount, 1)
    }

    func testBreakingIterationClosesTheNativeSession() async throws {
        let probe = StreamProbe(events: [
            .token("one", index: 0),
            .token("two", index: 1),
        ])

        try await consumeOne(probe.stream())

        XCTAssertEqual(probe.pullCount, 1)
        XCTAssertEqual(probe.closeCount, 1)
    }

    func testCancellationClosesAnInFlightPull() async throws {
        let probe = StreamProbe(events: [], blockPullUntilClosed: true)
        let stream = probe.stream()
        let task = Task { () throws -> XybridStreamToken? in
            var iterator = stream.makeAsyncIterator()
            return try await iterator.next()
        }

        XCTAssertTrue(probe.waitUntilPullStarts(timeout: 2))
        task.cancel()

        let result = try await task.value
        XCTAssertNil(result)
        XCTAssertEqual(probe.closeCount, 1)
    }

    private func consumeOne(_ stream: XybridTokenStream) async throws {
        for try await token in stream {
            XCTAssertEqual(token.token, "one")
            break
        }
    }
}

private final class StreamProbe: @unchecked Sendable {
    private let condition = NSCondition()
    private var events: [XybridStreamEvent]
    private let blockPullUntilClosed: Bool
    private var starts = 0
    private var pulls = 0
    private var closes = 0
    private var pullStarted = false
    private var closed = false

    init(events: [XybridStreamEvent], blockPullUntilClosed: Bool = false) {
        self.events = events
        self.blockPullUntilClosed = blockPullUntilClosed
    }

    var startCount: Int { read { starts } }
    var pullCount: Int { read { pulls } }
    var closeCount: Int { read { closes } }

    func stream() -> XybridTokenStream {
        XybridTokenStream(
            start: { self.start() },
            next: { try self.next(streamId: $0) },
            close: { self.close(streamId: $0) }
        )
    }

    func waitUntilPullStarts(timeout: TimeInterval) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(timeout)
        while !pullStarted {
            if !condition.wait(until: deadline) { return false }
        }
        return true
    }

    private func start() -> UInt64 {
        condition.lock()
        starts += 1
        condition.unlock()
        return 7
    }

    private func next(streamId: UInt64) throws -> XybridStreamEvent {
        guard streamId == 7 else {
            throw XybridError.inferenceError(message: "unexpected stream id")
        }
        condition.lock()
        pulls += 1
        pullStarted = true
        condition.broadcast()
        if blockPullUntilClosed {
            while !closed {
                condition.wait()
            }
            condition.unlock()
            throw XybridError.inferenceError(message: "stream closed")
        }
        guard !events.isEmpty else {
            condition.unlock()
            throw XybridError.inferenceError(message: "unexpected pull")
        }
        let event = events.removeFirst()
        condition.unlock()
        return event
    }

    private func close(streamId: UInt64) {
        guard streamId == 7 else { return }
        condition.lock()
        if !closed {
            closes += 1
            closed = true
        }
        condition.broadcast()
        condition.unlock()
    }

    private func read<T>(_ value: () -> T) -> T {
        condition.lock()
        defer { condition.unlock() }
        return value()
    }
}

private extension XybridStreamEvent {
    static func token(_ text: String, index: UInt64) -> Self {
        Self(
            kind: .token,
            token: XybridStreamToken(
                token: text,
                tokenId: nil,
                index: index,
                cumulativeText: text,
                finishReason: nil,
                toolCalls: [],
                rawText: nil
            )
        )
    }

    static var complete: Self {
        Self(kind: .complete, token: nil)
    }
}
