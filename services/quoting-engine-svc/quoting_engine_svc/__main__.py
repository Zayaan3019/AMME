"""Entrypoint: ``python -m quoting_engine_svc``."""

from __future__ import annotations

import os

from quoting_engine_svc.service import QuotingEngineService


def main() -> None:
    service = QuotingEngineService()
    service.serve(port=int(os.getenv("AMME_METRICS_PORT", "8004")))


if __name__ == "__main__":
    main()
