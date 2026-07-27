# syntax=docker/dockerfile:1
# DeepLOB inference. Hand-written rather than generated, for one reason: torch
# must come from the **CPU wheel index**.
#
# `pip install torch` on Linux resolves to the CUDA build by default, which
# pulls nvidia_cudnn_cu13 (366 MB), cuda_toolkit and friends -- several GB of
# image for a service the spec requires to run without a GPU. Pinning
# download.pytorch.org/whl/cpu keeps the image small and matches how inference
# actually runs here: single-threaded CPU, ONNX Runtime.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch in its own layer: largest and least frequently changed
# dependency, so it stays cached across rebuilds.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir onnxruntime

COPY shared/ /build/shared/
RUN pip install --no-cache-dir /build/shared

COPY services/deeplob-inference-svc/ /build/service/
# --no-deps: the service pyproject declares torch/onnxruntime for local dev, but
# both are already installed above from the CPU index. Without this pip would
# re-resolve them and drag the CUDA build back in.
RUN pip install --no-cache-dir --no-deps /build/service

# Runtime stage: no compilers, no build context, non-root.
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN groupadd --system amme && useradd --system --gid amme --home /app amme
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
RUN mkdir -p /app/models /app/artifacts && chown -R amme:amme /app
USER amme

EXPOSE 8003

# Health is an application-level assertion (model loaded, p99 inside budget),
# not just "the port is open".
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=4 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8003/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "deeplob_inference_svc"]
