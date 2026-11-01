"""A tiny event bus so the agent core never imports the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Event:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


Emitter = Callable[[Event], None]


def noop_emitter(event: Event) -> None:  # pragma: no cover - trivial
    return None


class EventBus:
    """Collects listeners; failing listeners never break the agent loop."""

    def __init__(self) -> None:
        self._listeners: list[Emitter] = []

    def subscribe(self, listener: Emitter) -> None:
        self._listeners.append(listener)

    def emit(self, kind: str, **data: Any) -> None:
        event = Event(kind, data)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # a broken renderer must not kill a task
                continue
