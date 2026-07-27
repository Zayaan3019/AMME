"""Entrypoint: ``python -m risk_svc``."""

from __future__ import annotations

import os

from risk_svc.service import RiskService


def main() -> None:
    service = RiskService()
    service.serve(port=int(os.getenv("AMME_METRICS_PORT", "8005")))


if __name__ == "__main__":
    main()
