"""Entrypoint: ``python -m ingestion_svc``."""

from __future__ import annotations

import os

from ingestion_svc.service import IngestionService


def main() -> None:
    service = IngestionService()
    service.serve(port=int(os.getenv("AMME_METRICS_PORT", "8001")))


if __name__ == "__main__":
    main()
