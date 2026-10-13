FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels '.[service]'

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --no-index --find-links /wheels 'graphmem[service]' \
    && rm -rf /wheels \
    && useradd --uid 10001 --create-home graphmem \
    && mkdir -p /var/lib/graphmem \
    && chown graphmem:graphmem /var/lib/graphmem
USER graphmem
EXPOSE 8090
VOLUME /var/lib/graphmem
ENTRYPOINT ["graphmem", "serve"]
CMD ["--data-dir", "/var/lib/graphmem", "--host", "0.0.0.0", "--port", "8090"]
