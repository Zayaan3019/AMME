"""Research artifact loading for the dashboard's evidence views.

The numbers on the Research tab are the ones `make ablation`, `make screen` and
`make train` actually wrote to ``artifacts/``. Nothing here is transcribed by
hand, because a hand-copied figure is a figure that goes stale the first time
the research is re-run and nobody notices.

Files are read on request, not cached at import: re-running the research while
the stack is up should be visible on a page refresh. They are small (the
largest is ~40KB) and the endpoint is not on any hot path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Latency budgets are measured by `pytest -m latency`, which asserts them in CI
# rather than reporting them. The assertion thresholds and the measured p50/p99
# are kept here so the UI can show the budget *and* whether it is met -- the
# figures come from docs/RESULTS.md "Inference latency" and the clock-corrected
# staleness study in ARCHITECTURE.md.
LATENCY_BUDGETS: list[dict[str, Any]] = [
    {
        "stage": "Binance ingest staleness",
        "detail": "clock-corrected; wire time to first touch",
        "p50_ms": 63.8,
        "p99_ms": 422.0,
        "budget_ms": None,
        "note": "No budget: bounded by the venue and the public internet, not by us.",
    },
    {
        "stage": "A-S quote computation",
        "detail": "closed-form reservation price + optimal spread",
        "p50_ms": 0.05,
        "p99_ms": None,
        "budget_ms": 0.05,
        "note": "Pure arithmetic on scalars; no allocation on the hot path.",
    },
    {
        "stage": "DeepLOB inference (ONNX, T=50)",
        "detail": "the backend actually served",
        "p50_ms": 2.83,
        "p99_ms": 4.04,
        "budget_ms": 5.0,
        "note": "Shipped. Halving the paper's lookback is what bought the headroom.",
    },
    {
        "stage": "DeepLOB inference (ONNX, T=100)",
        "detail": "the paper's lookback, ONNX",
        "p50_ms": 5.42,
        "p99_ms": 7.06,
        "budget_ms": 5.0,
        "note": "Rejected: exceeds budget at the paper's T.",
    },
    {
        "stage": "DeepLOB inference (TorchScript, T=100)",
        "detail": "the obvious first choice",
        "p50_ms": 11.44,
        "p99_ms": 15.51,
        "budget_ms": 5.0,
        "note": "Rejected: 2.2x slower than ONNX on the same graph (p99, T=100).",
    },
]


def _read_json(path: Path) -> Any | None:
    """Parse ``path``, or return None if it is missing or unreadable.

    A missing artifact is the normal state of a fresh clone -- the research has
    simply not been run yet. The UI renders that as an explicit "not yet
    generated, run `make ablation`" panel, which is more useful than an error
    and far more useful than a zero.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_markdown_table(text: str) -> dict[str, Any] | None:
    """Extract the first GitHub-flavoured table from ``text``.

    The venue screen is written as Markdown because its primary consumer is a
    human reading the repo. Rather than duplicating it as JSON -- two artifacts
    that can disagree -- the gateway parses the one file that exists.
    """
    rows: list[list[str]] = []
    header: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            if header:
                break  # table ended; ignore trailing prose
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # The alignment row (|---|---:|) separates header from body.
        if all(set(c) <= {"-", ":", " "} and "-" in c for c in cells):
            continue
        if not header:
            header = cells
        else:
            rows.append(cells)
    if not header or not rows:
        return None
    return {"columns": header, "rows": rows}


def load_research(artifacts_dir: Path) -> dict[str, Any]:
    """Collect every research artifact the dashboard renders.

    Each key is independently nullable: a repo that has run the ablation but not
    the venue screen shows one panel and an empty-state for the other.
    """
    ablation = _read_json(artifacts_dir / "ablation_multiseed.json")
    model = _read_json(artifacts_dir / "deeplob_metrics.json")
    venues_md = _read_text(artifacts_dir / "venue_screen.md")

    return {
        "ablation": ablation,
        "model": model,
        "venues": _parse_markdown_table(venues_md) if venues_md else None,
        "latency": LATENCY_BUDGETS,
        "available": {
            "ablation": ablation is not None,
            "model": model is not None,
            "venues": venues_md is not None,
        },
    }


__all__ = ["LATENCY_BUDGETS", "load_research"]
