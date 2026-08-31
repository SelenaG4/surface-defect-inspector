FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless still needs libglib2.0 at runtime on debian-slim.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Telemetry is opt-in at build time. The platform's CD pipeline builds with
# WITH_TELEMETRY=true for Azure; Render and local builds stay light and the
# service no-ops its tracing. See requirements-telemetry.txt.
ARG WITH_TELEMETRY=false

COPY requirements.txt requirements-telemetry.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_TELEMETRY" = "true" ]; then \
         pip install --no-cache-dir -r requirements-telemetry.txt; \
       fi

COPY app ./app
COPY models ./models
COPY data ./data

EXPOSE 8000

# Shell form so $PORT expands -- Render (and similar) assign their own PORT env
# var and scan for the app on THAT port. Falls back to 8000 for a plain
# `docker run`. (Lesson carried over from earlier projects in this portfolio.)
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
