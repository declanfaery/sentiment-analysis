from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from sentiment_analysis.models import TickerSignal


def write_signals_csv(signals: Iterable[TickerSignal], path: Path) -> None:
    rows = [signal.as_dict() for signal in signals]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_signals_json(signals: Iterable[TickerSignal], path: Path) -> None:
    rows = [signal.as_dict() for signal in signals]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")

