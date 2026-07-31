"""Minimal CSV training logger — no new dependency (no tensorboard/
wandb) for what's still an early-stage, single-machine project. One
row per training iteration; easy to load with pandas/Excel/a plotting
script later for loss curves once there's enough real training history
to make a curve worth looking at."""

import csv
import time
from pathlib import Path


class TrainingLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", newline="")
        self._writer: csv.DictWriter | None = None

    def log(self, **fields) -> None:
        row = {"timestamp": time.time(), **fields}
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            if self._file.tell() == 0:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
