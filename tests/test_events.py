"""Tests for isotope_core.events module."""

import asyncio

import pytest

from isotope_core.events import AgentEventStream, EventStream
from isotope_core.types import (
    AgentEndEvent,
    AgentStartEvent,
    MessageStartEvent,
    TextContent,
    UserMessage,
)

# =============================================================================
# EventStream Tests
# =============================================================================


class TestEventStreamInit:
    """Tests for EventStream initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        stream: EventStream[str, str] = EventStream()
        assert not stream.is_done

    def test_init_with_callbacks(self) -> None:
        """Test initialization with is_complete and extract_result."""
        stream: EventStream[str, str] = EventStream(
            is_complete=lambda e: e == "done",
            extract_result=lambda e: e.upper(),
        )
        assert not stream.is_done


class TestEventStreamPush:
    """Tests for EventStream.push()."""

    def test_push_queues_event(self) -> None:
        """Test that push queues the event."""
        stream: EventStream[str, str] = EventStream()
        stream.push("hello")
        assert not stream._queue.empty()

    def test_push_notifies_subscribers(self) -> None:
        """Test that push notifies all subscribers."""
        stream: EventStream[str, str] = EventStream()
        received: list[str] = []
        stream.subscribe(lambda e: received.append(e))
        stream.push("hello")
        assert received == ["hello"]

    def test_push_after_done_is_noop(self) -> None:
        """Test that push after done does nothing."""
        stream: EventStream[str, str] = EventStream()
        stream.end()
        stream.push("hello")
        # Queue should only have the None sentinel from end()
        assert stream._queue.qsize() == 1

    def test_push_with_completion_check(self) -> None:
        """Test push with is_complete callback."""
        stream: EventStream[str, str] = EventStream(
            is_complete=lambda e: e == "done",
            extract_result=lambda e: "FINISHED",
        )
        stream.push("hello")
        assert not stream.is_done
        stream.push("done")
        assert stream.is_done

    def test_push_completion_sets_result(self) -> None:
        """Test that completion via push sets the result."""
        stream: EventStream[str, str] = EventStream(
            is_complete=lambda e: e == "done",
            extract_result=lambda e: "RESULT",
        )
        stream.push("done")
        assert stream._final_result == "RESULT"
        assert stream._result_event.is_set()

    def test_push_subscriber_exception_suppressed(self) -> None:
        """Test that subscriber exceptions are suppressed."""
        stream: EventStream[str, str] = EventStream()

        def bad_callback(e: str) -> None:
            raise ValueError("oops")

        stream.subscribe(bad_callback)
        # Should not raise
        stream.push("hello")


class TestEventStreamEnd:
    """Tests for EventStream.end()."""

    def test_end_marks_done(self) -> None:
        """Test that end marks the stream as done."""
        stream: EventStream[str, str] = EventStream()
        stream.end()
        assert stream.is_done

    def test_end_with_result(self) -> None:
        """Test end with a result."""
        stream: EventStream[str, str] = EventStream()
        stream.end("my_result")
        assert stream._final_result == "my_result"
        assert stream._result_event.is_set()

    def test_end_without_result(self) -> None:
        """Test end without a result still sets the event."""
        stream: EventStream[str, str] = EventStream()
        stream.end()
        assert stream._final_result is None
        assert stream._result_event.is_set()

    def test_end_sends_sentinel(self) -> None:
        """Test that end sends None sentinel to queue."""
        stream: EventStream[str, str] = EventStream()
        stream.end()
        # Queue should contain None sentinel
        item = stream._queue.get_nowait()
        assert item is None

    def test_end_idempotent(self) -> None:
        """Test that calling end twice is a no-op."""
        stream: EventStream[str, str] = EventStream()
        stream.end("first")
        stream.end("second")
        assert stream._final_result == "first"


class TestEventStreamSubscribe:
    """Tests for EventStream.subscribe()."""

    def test_subscribe_returns_unsubscribe(self) -> None:
        """Test that subscribe returns an unsubscribe function."""
        stream: EventStream[str, str] = EventStream()
        received: list[str] = []
        unsub = stream.subscribe(lambda e: received.append(e))

        stream.push("before")
        unsub()
        stream.push("after")

        assert received == ["before"]

    def test_unsubscribe_idempotent(self) -> None:
        """Test that calling unsubscribe twice is safe."""
        stream: EventStream[str, str] = EventStream()
        unsub = stream.subscribe(lambda e: None)
        unsub()
        unsub()  # Should not raise


class TestEventStreamResult:
    """Tests for EventStream.result()."""

    async def test_result_waits_for_completion(self) -> None:
        """Test that result() waits for the stream to complete."""
        stream: EventStream[str, str] = EventStream()

        async def set_result() -> None:
            await asyncio.sleep(0.01)
            stream.end("done!")

        task = asyncio.create_task(set_result())
        result = await stream.result()
        assert result == "done!"
        await task

    async def test_result_returns_none_when_no_result(self) -> None:
        """Test result() returns None when no result is provided."""
        stream: EventStream[str, str] = EventStream()
        stream.end()
        result = await stream.result()
        assert result is None


class TestEventStreamAsyncIteration:
    """Tests for EventStream async iteration."""

    async def test_async_iteration(self) -> None:
        """Test async for iteration over events."""
        stream: EventStream[str, str] = EventStream()

        async def push_events() -> None:
            await asyncio.sleep(0.01)
            stream.push("a")
            stream.push("b")
            stream.push("c")
            stream.end()

        task = asyncio.create_task(push_events())
        collected: list[str] = []
        async for event in stream:
            collected.append(event)

        assert collected == ["a", "b", "c"]
        await task

    async def test_aiter_returns_self(self) -> None:
        """Test that __aiter__ returns self."""
        stream: EventStream[str, str] = EventStream()
        assert stream.__aiter__() is stream

    async def test_anext_raises_stop_on_done_and_empty(self) -> None:
        """Test that __anext__ raises StopAsyncIteration when done and empty."""
        stream: EventStream[str, str] = EventStream()
        stream.end()
        # Consume the None sentinel
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    async def test_anext_on_none_sentinel(self) -> None:
        """Test that __anext__ raises StopAsyncIteration on None sentinel."""
        stream: EventStream[str, str] = EventStream()
        stream.push("event")
        stream.end()
        # First call returns the event
        result = await stream.__anext__()
        assert result == "event"
        # Second call hits None sentinel
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()


class TestEventStreamIsDone:
    """Tests for EventStream.is_done property."""

    def test_is_done_false_initially(self) -> None:
        """Test that is_done is False initially."""
        stream: EventStream[str, str] = EventStream()
        assert stream.is_done is False

    def test_is_done_true_after_end(self) -> None:
        """Test that is_done is True after end()."""
        stream: EventStream[str, str] = EventStream()
        stream.end()
        assert stream.is_done is True


# =============================================================================
# AgentEventStream Tests
# =============================================================================


class TestAgentEventStream:
    """Tests for AgentEventStream."""

    def test_init(self) -> None:
        """Test initialization."""
        stream = AgentEventStream()
        assert not stream.is_done

    def test_agent_end_completes_stream(self) -> None:
        """Test that AgentEndEvent completes the stream."""
        stream = AgentEventStream()
        end_event = AgentEndEvent(messages=[], reason="completed")
        stream.push(end_event)
        assert stream.is_done

    async def test_agent_end_extracts_messages(self) -> None:
        """Test that AgentEndEvent extracts messages as result."""
        stream = AgentEventStream()
        msg = UserMessage(
            content=[TextContent(text="hello")],
            timestamp=1000,
        )
        end_event = AgentEndEvent(messages=[msg], reason="completed")
        stream.push(end_event)
        result = await stream.result()
        assert result is not None
        assert len(result) == 1

    def test_non_agent_end_does_not_complete(self) -> None:
        """Test that non-AgentEndEvent events don't complete the stream."""
        stream = AgentEventStream()
        start_event = AgentStartEvent()
        stream.push(start_event)
        assert not stream.is_done

    async def test_extract_result_non_agent_end_returns_empty(self) -> None:
        """Test that extract_result returns [] for non-AgentEndEvent."""
        stream = AgentEventStream()
        # Manually call the extract_result callback
        assert stream._extract_result is not None
        result = stream._extract_result(AgentStartEvent())
        assert result == []

    async def test_full_lifecycle(self) -> None:
        """Test full lifecycle: push events, iterate, get result."""
        stream = AgentEventStream()

        msg = UserMessage(content=[TextContent(text="hi")], timestamp=1000)
        events_to_push = [
            AgentStartEvent(),
            MessageStartEvent(message=msg),
            AgentEndEvent(messages=[msg], reason="completed"),
        ]

        async def push_all() -> None:
            await asyncio.sleep(0.01)
            for e in events_to_push:
                stream.push(e)
            stream.end()

        task = asyncio.create_task(push_all())
        collected = []
        async for event in stream:
            collected.append(event)

        assert len(collected) == 3
        result = await stream.result()
        assert result is not None
        assert len(result) == 1
        await task
