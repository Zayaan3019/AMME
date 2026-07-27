"""Entrypoint: ``python -m api_gateway_svc``."""

from __future__ import annotations

import os

from api_gateway_svc.service import ApiGatewayService


def main() -> None:
    service = ApiGatewayService()
    service.serve(port=int(os.getenv("AMME_METRICS_PORT", "8000")))


if __name__ == "__main__":
    main()
