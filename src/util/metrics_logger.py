"""
Dynamic CSV metrics logger for RL experiments.

Accumulates per-iteration metrics and writes them to a CSV file.
Columns are determined dynamically based on what each experiment logs,
so different experiment types (MADDPG, RLFD, RFT, IBMARL) can log
different sets of metrics without any hardcoded schema.
"""

import csv
from pathlib import Path


class MetricsLogger:

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []
        self._columns: list[str] = ["iteration", "group"]

    def log(self, iteration: int, group: str, **metrics) -> None:
        """
        Record one row of metrics for a given iteration and agent group.

        Any keyword argument becomes a column in the CSV. Experiments only
        need to pass the metrics they care about; absent columns are left
        blank for other experiment types.
        """
        row = {"iteration": iteration, "group": group}
        for key, value in metrics.items():
            if key not in self._columns:
                self._columns.append(key)
            if value is not None:
                row[key] = value
        self._rows.append(row)

    def save(self) -> None:
        """Write all accumulated rows to CSV (full rewrite)."""
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._columns, extrasaction="ignore"
            )
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def get_values(self, column: str, group: str | None = None) -> list:
        """
        Retrieve all logged values for *column*, optionally filtered by group.
        Missing entries are returned as ``None``.
        """
        values = []
        for row in self._rows:
            if group is not None and row.get("group") != group:
                continue
            values.append(row.get(column))
        return values

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def rows(self) -> list[dict]:
        return list(self._rows)
