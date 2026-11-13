"""Make the service packages importable from tests.

Services are independent distributions in the deployed topology and are not
installed into the dev venv, so their packages are not on ``sys.path`` by
default. Tests that exercise service-level code (the replay merger, the
Binance URL builder) add the relevant directory here rather than each doing
its own ``sys.path`` surgery inline.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERVICES = Path(__file__).resolve().parent.parent / "services"

for _svc in (
    "backtest-svc", "ingestion-svc", "deeplob-inference-svc", "api-gateway-svc",
    "risk-svc", "execution-sim-svc",
):
    _path = _SERVICES / _svc
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# The training package (``amme_train``) for the live-recording dataset tests.
_TRAIN = Path(__file__).resolve().parent.parent / "train"
if _TRAIN.is_dir() and str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))
