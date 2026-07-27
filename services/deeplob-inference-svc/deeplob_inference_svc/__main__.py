"""Entrypoint: ``python -m deeplob_inference_svc``."""

from __future__ import annotations

import os

from deeplob_inference_svc.service import DeepLOBInferenceService


def main() -> None:
    service = DeepLOBInferenceService()
    service.serve(port=int(os.getenv("AMME_METRICS_PORT", "8003")))


if __name__ == "__main__":
    main()
