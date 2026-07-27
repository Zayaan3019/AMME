"""Entrypoint: ``python -m execution_sim_svc``."""

from __future__ import annotations

import os

from execution_sim_svc.service import ExecutionSimService


def main() -> None:
    service = ExecutionSimService()
    service.serve(port=int(os.getenv("AMME_METRICS_PORT", "8006")))


if __name__ == "__main__":
    main()
