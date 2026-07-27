"""Entrypoint: ``python -m feature_svc``."""

from __future__ import annotations

import os

from feature_svc.service import FeatureService


def main() -> None:
    service = FeatureService()
    service.serve(port=int(os.getenv("AMME_METRICS_PORT", "8002")))


if __name__ == "__main__":
    main()
