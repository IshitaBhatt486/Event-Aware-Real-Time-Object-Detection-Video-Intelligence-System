"""Storage adapters for structured detection events."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class JsonlEventLogger:
    """Append one JSON event per line and flush it to disk immediately."""

    def __init__(self, path: str | Path = "events.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        """Append an event without altering its structure."""
        line = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        """Flush and close the event file."""
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
