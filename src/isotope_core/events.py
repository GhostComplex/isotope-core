"""Async event stream implementation.

This module provides the EventStream class for async iteration and
push-based consumption of events.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from isotope_core.types import AgentEvent, Message

T = TypeVar("T")  # Event type
R = TypeVar("R")  # Final result type


class EventStream(Generic[T, R]):
    """Async event stream supporting both pull and push-based consumption.

    This class provides:
    - Async iteration via `async for event in stream:`
    - Push-based consumption via `subscribe(callback)`
    - Access to the final result via `result()`
    - Proper cleanup on abort

    Example:
        stream = EventStream(
            is_complete=lambda e: e.type == "done",
            extract_result=lambda e: e.final_value,
        )

        # Producer pushes events
        stream.push(event1)
        stream.push(event2)
        stream.end(final_result)

        # Consumer iterates
        async for event in stream:
            print(event)

        # Get final result
        result = await stream.result()
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool] | None = None,
        extract_result: Callable[[T], R] | None = None,
    ):
        """Initialize the EventStream.

        Args:
            is_complete: Function to determine if an event signals completion.
            extract_result: Function to extract the final result from a completion event.
        """
        self._queue: asyncio.Queue[T | None] = asyncio.Queue()
        self._subscribers: list[Callable[[T], None]] = []
        self._done = False
        self._final_result: R | None = None
        self._result_event = asyncio.Event()
        self._is_complete = is_complete
        self._extract_result = extract_result

    def push(self, event: T) -> None:
        """Push an event to the stream.

        This delivers the event to:
        1. Any async iterators waiting
        2. All subscribed callbacks

        Args:
            event: The event to push.
        """
        if self._done:
            return

        # Check if this event completes the stream
        if self._is_complete and self._is_complete(event):
            self._done = True
            if self._extract_result:
                self._final_result = self._extract_result(event)
                self._result_event.set()

        # Queue for async iteration
        self._queue.put_nowait(event)

        # Notify subscribers
        for callback in self._subscribers:
            with contextlib.suppress(Exception):
                callback(event)

    def end(self, result: R | None = None) -> None:
        """End the stream, optionally with a final result.

        Args:
            result: The final result (used if no completion event was pushed).
        """
        if self._done:
            return

        self._done = True
        if result is not None:
            self._final_result = result
        self._result_event.set()

        # Signal end to any waiting iterators
        self._queue.put_nowait(None)

    def subscribe(self, callback: Callable[[T], None]) -> Callable[[], None]:
        """Subscribe to events via callback.

        Args:
            callback: Function called for each event.

        Returns:
            An unsubscribe function.
        """
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    async def result(self) -> R | None:
        """Get the final result of the stream.

        Waits for the stream to complete if necessary.

        Returns:
            The final result, or None if no result was provided.
        """
        await self._result_event.wait()
        return self._final_result

    def __aiter__(self) -> AsyncIterator[T]:
        """Return async iterator for the stream."""
        return self

    async def __anext__(self) -> T:
        """Get the next event from the stream."""
        if self._done and self._queue.empty():
            raise StopAsyncIteration

        event = await self._queue.get()
        if event is None:
            raise StopAsyncIteration

        return event

    @property
    def is_done(self) -> bool:
        """Check if the stream has completed."""
        return self._done


class AgentEventStream(EventStream["AgentEvent", "list[Message]"]):
    """Specialized event stream for agent events.

    This extends EventStream with agent-specific functionality.
    """

    def __init__(self) -> None:
        """Initialize an AgentEventStream."""
        from isotope_core.types import AgentEndEvent

        def is_complete(e: AgentEvent) -> bool:
            return e.type == "agent_end"

        def extract_result(e: AgentEvent) -> list[Message]:
            if isinstance(e, AgentEndEvent):
                return e.messages
            return []

        super().__init__(
            is_complete=is_complete,
            extract_result=extract_result,
        )
