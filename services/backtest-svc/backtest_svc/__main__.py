"""Entrypoint: ``python -m backtest_svc``."""

from __future__ import annotations

import os

from backtest_svc.service import BacktestService


def main() -> None:
    service = BacktestService()
    service.serve(port=int(os.getenv("AMME_METRICS_PORT", "8007")))


if __name__ == "__main__":
    main()
