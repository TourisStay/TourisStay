FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libraries: none required. On python:3.11-slim the manylinux wheels for
# shapely (GEOS), pyproj (PROJ), pyogrio (GDAL) and psycopg2-binary (libpq)
# bundle their native libraries, so no apt packages are needed to install or run.
# If a future dependency needs one, add it here with a comment naming the package
# it is for, and clean up in the same layer:
#   RUN apt-get update \
#       && apt-get install -y --no-install-recommends <pkg> \
#       && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the etl/ folder (build context) into /app. Raw data is NOT copied;
# it is mounted at runtime at /data. DATABASE_URL is supplied at runtime.
COPY . .

# Non-root user: reads /app, writes /data.
RUN useradd --create-home --uid 1000 etl \
    && mkdir -p /data \
    && chown -R etl:etl /data \
    && chmod -R a+rX /app

USER etl

CMD ["sh", "-c", "python ingest.py && python clean.py && python load.py"]
