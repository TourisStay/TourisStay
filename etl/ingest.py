"""Raw integration of the manually downloaded Inside Airbnb files.

Reads   <DATA_DIR>/raw/<country>/<city>/{listings,calendar,reviews}.csv[.gz]
        and neighbourhoods.geojson
Writes  <DATA_DIR>/integrated/{listings,calendar,reviews,neighbourhoods}.parquet
        and ingest_summary.json

No values are changed: every CSV is read as text (dtype=str,
keep_default_na=False) and the tables are stacked across cities. The only
added columns are `country` and `city`, taken from the folder names.
Cleaning happens later, in a separate script.

The CSV tables are streamed to Parquet one chunk at a time instead of being
concatenated in memory, because the reviews alone are well over 10 GB as text.
The output matches pd.concat(..., ignore_index=True, sort=False): columns are
the union of all cities' columns in first-seen order, and columns a city lacks
are filled with empty strings.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CSV_TABLES = ("listings", "calendar", "reviews")
GEO_TABLE = "neighbourhoods"
LINEAGE_COLUMNS = ("country", "city")
CHUNK_ROWS = 200_000

READ_CSV_KWARGS = dict(dtype=str, keep_default_na=False, encoding="utf-8")


def data_dir() -> Path:
    """DATA_DIR from the environment, else <repo root>/data.

    The repo root is the parent of the folder holding this script, so the
    default is ./data locally (etl/ingest.py -> ./data) and /data inside the
    container (/app/ingest.py -> /data).
    """
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env).resolve()
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def discover_cities(raw_dir: Path) -> list[tuple[str, str, Path]]:
    cities = []
    for country_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for city_dir in sorted(p for p in country_dir.iterdir() if p.is_dir()):
            cities.append((country_dir.name, city_dir.name, city_dir))
    return cities


def find_csv(city_dir: Path, table: str) -> Path | None:
    """Return <table>.csv or <table>.csv.gz. If both exist, .csv wins."""
    plain = city_dir / f"{table}.csv"
    gz = city_dir / f"{table}.csv.gz"
    if plain.is_file() and gz.is_file():
        print(f"WARNING {table:<14} {city_dir.parent.name}/{city_dir.name}: "
              f"both {plain.name} and {gz.name} exist, using {plain.name}")
    if plain.is_file():
        return plain
    if gz.is_file():
        return gz
    return None


def warn_skip(table: str, label: str, reason: str, skipped: list[dict]) -> None:
    print(f"WARNING {table:<14} {label}: skipped ({reason})")
    skipped.append({"city": label, "reason": reason})


def check_no_lineage_clash(columns, table: str, label: str) -> None:
    clash = [c for c in LINEAGE_COLUMNS if c in columns]
    if clash:
        raise RuntimeError(
            f"{table} for {label} already has column(s) {clash}; refusing to "
            "overwrite them with lineage values")


def ingest_csv_table(table: str, cities, out_dir: Path) -> dict:
    skipped: list[dict] = []
    sources: list[tuple[str, str, Path]] = []
    union: list[str] = []
    seen: set[str] = set()

    # Pass 1: read only the headers to build the column union.
    for country, city, city_dir in cities:
        label = f"{country}/{city}"
        path = find_csv(city_dir, table)
        if path is None:
            warn_skip(table, label, f"no {table}.csv or {table}.csv.gz", skipped)
            continue
        try:
            header = pd.read_csv(path, nrows=0, **READ_CSV_KWARGS).columns
        except pd.errors.EmptyDataError:
            warn_skip(table, label, f"{path.name} is empty", skipped)
            continue
        check_no_lineage_clash(header, table, label)
        for col in header:
            if col not in seen:
                seen.add(col)
                union.append(col)
        sources.append((country, city, path))

    columns = union + list(LINEAGE_COLUMNS)
    schema = pa.schema([pa.field(c, pa.string()) for c in columns])
    out_path = out_dir / f"{table}.parquet"
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    # Pass 2: stream every city into one Parquet file.
    total = 0
    included: list[str] = []
    with pq.ParquetWriter(tmp_path, schema) as writer:
        for country, city, path in sources:
            label = f"{country}/{city}"
            city_rows = 0
            for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, **READ_CSV_KWARGS):
                chunk = chunk.reindex(columns=union, fill_value="")
                chunk["country"] = country
                chunk["city"] = city
                writer.write_table(
                    pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
                city_rows += len(chunk)
            print(f"{table:<14} {label:<36} {city_rows:>12,} rows")
            total += city_rows
            included.append(label)
    os.replace(tmp_path, out_path)

    return {
        "cities_included": len(included),
        "rows": total,
        "columns": len(columns),
        "cities": included,
        "skipped": skipped,
    }


def ingest_neighbourhoods(cities, out_dir: Path) -> dict:
    table = GEO_TABLE
    skipped: list[dict] = []
    frames: list[gpd.GeoDataFrame] = []
    included: list[str] = []

    for country, city, city_dir in cities:
        label = f"{country}/{city}"
        path = city_dir / f"{table}.geojson"
        if not path.is_file():
            warn_skip(table, label, f"no {table}.geojson", skipped)
            continue
        gdf = gpd.read_file(path)
        check_no_lineage_clash(gdf.columns, table, label)
        # GeoJSON (RFC 7946) is always WGS84. GDAL labels files with 3D
        # coordinates as EPSG:4979 (WGS84 3D), whose horizontal axes are
        # identical to EPSG:4326. Relabel only; coordinates (including Z)
        # are not transformed.
        if gdf.crs is not None and gdf.crs.to_epsg() == 4979:
            print(f"INFO    {table:<14} {label}: CRS read as EPSG:4979 (3D GeoJSON), "
                  "labelled EPSG:4326, coordinates unchanged")
            gdf = gdf.set_crs(4326, allow_override=True)
        gdf["country"] = country
        gdf["city"] = city
        print(f"{table:<14} {label:<36} {len(gdf):>12,} rows")
        frames.append(gdf)
        included.append(label)

    if not frames:
        raise RuntimeError("no neighbourhoods.geojson found in any city")

    crs_set = {str(f.crs) for f in frames}
    if len(crs_set) > 1:
        raise RuntimeError(f"neighbourhoods files use different CRS: {sorted(crs_set)}")

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True, sort=False),
        geometry=frames[0].geometry.name,
        crs=frames[0].crs,
    )
    out_path = out_dir / f"{table}.parquet"
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)

    return {
        "cities_included": len(included),
        "rows": len(combined),
        "columns": len(combined.columns),
        "cities": included,
        "skipped": skipped,
    }


def main() -> int:
    base = data_dir()
    raw_dir = base / "raw"
    out_dir = base / "integrated"
    if not raw_dir.is_dir():
        print(f"ERROR raw data folder not found: {raw_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    cities = discover_cities(raw_dir)
    print(f"Found {len(cities)} city folders under {raw_dir}")

    summary: dict[str, dict] = {}
    for table in CSV_TABLES:
        summary[table] = ingest_csv_table(table, cities, out_dir)
    summary[GEO_TABLE] = ingest_neighbourhoods(cities, out_dir)

    with open(out_dir / "ingest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\nIntegrated totals")
    for table, info in summary.items():
        print(f"{table:<14} {info['rows']:>12,} rows  {info['columns']:>4} columns  "
              f"{info['cities_included']} cities  {len(info['skipped'])} skipped")
    print(f"Output written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
