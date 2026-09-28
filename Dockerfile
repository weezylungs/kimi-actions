FROM python:3.12-slim

WORKDIR /app

# Cache bust - update this to force rebuild
ARG CACHE_BUST=20260117_v1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for building kimi-sdk and kimi-agent-sdk
RUN pip install --no-cache-dir uv==0.7.22

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep Semgrep in an isolated tool environment so its mcp dependency cannot
# mutate kimi-agent-sdk's runtime dependency graph.
ENV UV_TOOL_BIN_DIR=/usr/local/bin
RUN uv tool install semgrep || echo "Semgrep installation skipped"

# Fail the image build if the Action runtime has incompatible dependencies.
RUN pip check

# Configure git for agent operations
RUN git config --global user.name "Kimi Bot" && \
    git config --global user.email "kimi@moonshot.cn"

COPY src/ ./src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
