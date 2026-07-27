# syntax=docker/dockerfile:1
# Build stage: compile wheels for shared + this service.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1     PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Shared package first: it changes far less often than service code, so this
# layer stays cached across most rebuilds.
COPY shared/ /build/shared/
RUN pip install --no-cache-dir /build/shared

COPY services/api-gateway-svc/ /build/service/
RUN pip install --no-cache-dir /build/service

# Runtime stage: no compilers, no build context, non-root.
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH"     PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1

RUN groupadd --system amme && useradd --system --gid amme --home /app amme
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
RUN mkdir -p /app/models /app/artifacts && chown -R amme:amme /app
USER amme

EXPOSE 8000

# Health is an application-level assertion (book synced, P&L identity holds),
# not just "the port is open" -- see each service's health checks.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=4     CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "api_gateway_svc"]
