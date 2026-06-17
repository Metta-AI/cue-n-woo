FROM docker.io/library/python:3.12-slim

# Runtime deps for the game referee and the bundled player policies. Bedrock
# serves the judge/scorer remotely, so torch/flas are intentionally absent and
# the image stays small.
RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    "uvicorn[standard]==0.34.2" \
    websockets==15.0.1 \
    cryptography==48.0.0 \
    boto3==1.42.6

ENV PYTHONPATH=/app
WORKDIR /app

# Imports are rooted at `v2.` (e.g. `from v2 import signing`,
# `from v2.coworld.harness import ...`), so mirror that tree under /app.
RUN mkdir -p /app/v2 && touch /app/v2/__init__.py
COPY signing.py /app/v2/signing.py
COPY coworld /app/v2/coworld

# Default command is the game referee; players override `run` in the manifest.
CMD ["python", "-m", "v2.coworld.game"]
