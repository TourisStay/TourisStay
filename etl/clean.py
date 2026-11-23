"""Clean the integrated Inside Airbnb tables.

Reads   <DATA_DIR>/integrated/{listings,calendar,reviews,neighbourhoods}.parquet
Writes  <DATA_DIR>/cleaned/{listings,calendar,reviews,neighbourhoods}.parquet
        <DATA_DIR>/cleaned/cleaning_report.json and cleaning_report.md

Input files are never modified. Every input column is text, exactly as
published; empty strings are treated as nulls.

Listings fit in memory and are cleaned with pandas plus pyarrow compute.
Calendar (~250M rows) and reviews (~30M rows, long text) are cleaned one
city at a time with pyarrow, using the per-city row groups that ingest.py
writes, and streamed to the output file.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ingest import data_dir

LINEAGE = ("country", "city")

# ---------------------------------------------------------------------------
# Parsing helpers (pyarrow). Each takes a string array whose empty values are
# already null, and returns the parsed array. Non-empty values that fail to
# parse become null and are recorded in `bad`.
# ---------------------------------------------------------------------------

CURRENCY_OR_SPACE = r"[\p{Sc}\s]+"      # any Unicode currency symbol, any whitespace
PRICE_OK = r"^(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$"   # 1200 / 1,200 / 1,200.00 / 83.5
INT_OK = r"^-?\d{1,18}$"
INT19 = r"^-?\d{19}$"
INT64_MAX = "9223372036854775807"
FLOAT_OK = r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$"
NULL_STR = pa.scalar(None, pa.string())


class BadValues:
    """Counts unparseable values per column, keeping all distinct values."""

    def __init__(self):
        self.by_col: dict[str, Counter] = defaultdict(Counter)

    def add(self, col: str, values: pa.ChunkedArray | pa.Array) -> None:
        if len(values) == 0:
            return
        for item in pc.value_counts(values).to_pylist():
            self.by_col[col][item["values"]] += item["counts"]

    def report(self) -> dict:
        out = {}
        for col in sorted(self.by_col):
            c = self.by_col[col]
            out[col] = {
                "count": sum(c.values()),
                "distinct": len(c),
                "examples": [{"value": v, "count": n}
                             for v, n in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:10]],
            }
        return out


def empty_to_null(arr, strip: bool = False):
    if strip:
        arr = pc.utf8_trim_whitespace(arr)
    return pc.if_else(pc.equal(arr, ""), NULL_STR, arr)


def _finish(arr, ok, cleaned, target, col, bad):
    ok = pc.fill_null(ok, False)
    bad.add(col, pc.filter(arr, pc.and_kleene(pc.is_valid(arr), pc.invert(ok))))
    return pc.cast(pc.if_else(ok, cleaned, NULL_STR), target)


def parse_price(arr, col, bad):
    s = pc.replace_substring_regex(arr, CURRENCY_OR_SPACE, "")
    ok = pc.match_substring_regex(s, PRICE_OK)
    return _finish(arr, ok, pc.replace_substring(s, ",", ""), pa.float64(), col, bad)


def parse_int(arr, col, bad):
    # Up to 18 digits always fits int64; 19-digit values (Airbnb listing ids
    # are 19 digits) fit if not above int64 max. Equal-length digit strings
    # compare in numeric order.
    ok = pc.or_kleene(
        pc.match_substring_regex(arr, INT_OK),
        pc.and_kleene(pc.match_substring_regex(arr, INT19),
                      pc.less_equal(pc.utf8_ltrim(arr, characters="-"), INT64_MAX)))
    return _finish(arr, ok, arr, pa.int64(), col, bad)


def parse_float(arr, col, bad):
    return _finish(arr, pc.match_substring_regex(arr, FLOAT_OK), arr, pa.float64(), col, bad)


def parse_date(arr, col, bad):
    ts = pc.strptime(arr, format="%Y-%m-%d", unit="s", error_is_null=True)
    bad.add(col, pc.filter(arr, pc.and_kleene(pc.is_valid(arr), pc.is_null(ts))))
    return pc.cast(ts, pa.date32())


def parse_bool_tf(arr, col, bad):
    is_t, is_f = pc.equal(arr, "t"), pc.equal(arr, "f")
    bad.add(col, pc.filter(arr, pc.and_kleene(pc.is_valid(arr),
                                              pc.invert(pc.or_(is_t, is_f)))))
    return pc.if_else(is_t, True, pc.if_else(is_f, False, pa.scalar(None, pa.bool_())))


PARSERS = {"price": parse_price, "int": parse_int, "float": parse_float,
           "date": parse_date, "bool_tf": parse_bool_tf}


def apply_parsers(t: pa.Table, kinds: dict[str, str], bad: BadValues) -> pa.Table:
    for col, kind in kinds.items():
        if col in t.column_names:
            i = t.column_names.index(col)
            t = t.set_column(i, col, PARSERS[kind](t[col], col, bad))
    return t


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def raw_null_counts(t: pa.Table) -> dict[str, int]:
    """Nulls in an all-text input table: null or empty string."""
    return {c: int(t[c].null_count + (pc.sum(pc.equal(t[c], "")).as_py() or 0))
            if pa.types.is_string(t[c].type) else int(t[c].null_count)
            for c in t.column_names}


def null_counts(t: pa.Table) -> dict[str, int]:
    return {c: int(t[c].null_count) for c in t.column_names}


def city_label(country: str, city: str) -> str:
    return f"{country}/{city}"


def city_counts(t: pa.Table) -> Counter:
    keys = pc.binary_join_element_wise(t["country"], t["city"], "/")
    return Counter({d["values"]: d["counts"] for d in pc.value_counts(keys).to_pylist()})


def required_mask(t: pa.Table, required: list[str]):
    """Return (row-drop mask, per-field null counts) for fields that exist."""
    by_field, mask = {}, None
    for col in required:
        if col not in t.column_names:
            continue
        m = pc.is_null(t[col])
        by_field[col] = int(pc.sum(m).as_py() or 0)
        mask = m if mask is None else pc.or_(mask, m)
    if mask is None:
        mask = pa.array([False] * t.num_rows)
    return mask, by_field


def city_row_groups(path: Path) -> list[tuple[str, str, list[int]]]:
    """Map each (country, city) to its row groups using column statistics.

    ingest.py writes each city in its own chunks, so every row group holds a
    single city. Falls back to reading the columns if that ever stops holding.
    """
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    ci, ti = names.index("country"), names.index("city")
    groups: dict[tuple[str, str], list[int]] = {}
    for i in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(i)
        sc, st = rg.column(ci).statistics, rg.column(ti).statistics
        if not (sc and st and sc.has_min_max and st.has_min_max
                and sc.min == sc.max and st.min == st.max):
            raise RuntimeError(f"{path.name}: row group {i} is not single-city; "
                               "re-run ingest.py")
        groups.setdefault((sc.min, st.min), []).append(i)
    return [(c, t, g) for (c, t), g in groups.items()]


def atomic_write_table(t: pa.Table, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(t, tmp)
    os.replace(tmp, path)


def check_rows(info: dict) -> None:
    removed = sum(s["removed"] for s in info["steps"])
    info["rows_add_up"] = info["rows_before"] - removed == info["rows_after"]
    if not info["rows_add_up"]:
        raise RuntimeError(f"row arithmetic does not add up: {info['rows_before']} - "
                           f"{removed} != {info['rows_after']}")


def per_city_table(before: Counter, after: Counter) -> dict:
    return {k: {"before": before.get(k, 0), "after": after.get(k, 0),
                "removed": before.get(k, 0) - after.get(k, 0)}
            for k in sorted(set(before) | set(after))}


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------

LISTING_PARSE = {"price": "price", "id": "int", "host_id": "int",
                 "minimum_nights": "int", "number_of_reviews": "int",
                 "latitude": "float", "longitude": "float",
                 "reviews_per_month": "float"}
LISTING_REQUIRED = ["id", "country", "city", "price"]

# License rules, applied in this order to the lower-cased, stripped text.
LICENSE_RULES = [
    ("Exempt", "states no license is needed",
     r"licen[cs]e not (?:needed|required)|no licen[cs]e (?:needed|required)|not required"),
    ("Unlicensed", "pending / applied / unlicensed",
     r"pending|pendiente|en tr.mite|applied|application|in process|in progress|awaiting"
     r"|unlicen[cs]ed|not licen[cs]ed|no licen[cs]e|sin licencia|without (?:a )?licen[cs]e"),
    ("Licensed", "contains a registration number (has a digit)", r"\d"),
    ("Exempt", "says exempt (exempt, exento, bail mobilite)",
     r"exempt|exent[oa]|bail mobilit"),
    ("Licensed", "says licensed or registered", r"\blicen[cs]ed\b|\bregistered\b"),
]
LICENSE_VALUES = ("Licensed", "Unlicensed", "Exempt", "Unknown")
HOST_VALUES = ("Individual", "Commercial", "Unknown")


def classify_license(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (category, rule description) for each raw license value."""
    text = raw.fillna("").str.lower()
    cat = pd.Series("Unknown", index=raw.index, dtype=object)
    rule = pd.Series("empty", index=raw.index, dtype=object)
    rule[text != ""] = "no rule matched"
    todo = text != ""
    for category, desc, pattern in LICENSE_RULES:
        hit = todo & text.str.contains(pattern, regex=True)
        cat[hit], rule[hit] = category, desc
        todo &= ~hit
    return cat, rule


def value_shape(v: str) -> str:
    return re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", v))


def license_report(raw: pd.Series, cat: pd.Series, rule: pd.Series) -> dict:
    df = pd.DataFrame({"raw": raw.fillna(""), "cat": cat, "rule": rule})
    out = {"categories": {k: int((cat == k).sum()) for k in LICENSE_VALUES},
           "rules": [], "by_category": {}}
    for (c, r), n in df.groupby(["cat", "rule"]).size().items():
        out["rules"].append({"category": c, "rule": r, "rows": int(n)})
    for c in LICENSE_VALUES:
        sub = df[df.cat == c]
        has_digit = sub.raw.str.contains(r"\d")
        text_vals = sub.raw[~has_digit].value_counts()
        shapes = sub.raw[has_digit].map(value_shape)
        shape_counts = shapes.value_counts()
        example = sub.raw[has_digit].groupby(shapes).first()
        out["by_category"][c] = {
            "rows": int(len(sub)),
            "distinct_raw_values": int(sub.raw.nunique()),
            # Every raw value without a digit, in full.
            "text_values": [{"value": v, "count": int(n)} for v, n in text_vals.items()],
            # Values with digits (registration numbers) grouped by shape:
            # digit -> 9, letter -> A.
            "numbered_value_shapes_top50": [
                {"shape": s, "count": int(n), "example": example[s]}
                for s, n in shape_counts.head(50).items()],
            "numbered_values_distinct_shapes": int(len(shape_counts)),
        }
    return out


def clean_listings(src: Path, dst: Path, problems: list[str]) -> dict:
    t = pq.read_table(src)
    info: dict = {"rows_before": t.num_rows, "steps": []}
    info["nulls_before"] = raw_null_counts(t)
    before_city = city_counts(t)
    df = t.to_pandas()
    del t

    replacement = {c: int(df[c].str.contains("\ufffd", regex=False).sum())
                   for c in df.columns}
    replacement = {c: n for c, n in replacement.items() if n}
    if replacement:
        problems.append(
            "listings: some text contains the Unicode replacement character (U+FFFD), "
            "meaning characters were already lost in the source files; they cannot be "
            f"recovered. Rows affected per column: {replacement}")

    # 1. Exact duplicate rows, then duplicate (country, city, id) keys.
    n = len(df)
    df = df.drop_duplicates(keep="first")
    info["steps"].append({"step": "exact_duplicate_rows", "removed": n - len(df)})
    n = len(df)
    key_id = df["id"].str.strip()
    dup = pd.concat([df["country"], df["city"], key_id], axis=1).duplicated(keep="first")
    df = df[~dup]
    info["steps"].append({"step": "duplicate_country_city_id_keys",
                          "removed": n - len(df),
                          "note": "first occurrence kept"})

    # 2. Host listing counts, from the de-duplicated integrated data, before
    #    any rows are dropped for missing values.
    hid = df["host_id"].str.strip()
    hcount = df.groupby([df["country"], df["city"], hid])["id"].transform("size")
    hcount = hcount.where(hid != "")
    host_cat = pd.Series("Commercial", index=df.index, dtype=object)
    host_cat[hcount == 1] = "Individual"
    host_cat[hid == ""] = "Unknown"

    # 3. Strip text, empty -> null, parse types.
    t = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)
    del df
    t = pa.table({c: empty_to_null(t[c], strip=True) for c in t.column_names})
    price_empty = int(t["price"].null_count)
    bad = BadValues()
    t = apply_parsers(t, LISTING_PARSE, bad)

    # 4. Coordinates out of range -> null (both lat and lon).
    lat, lon = t["latitude"], t["longitude"]
    out_rng = pc.fill_null(pc.or_kleene(
        pc.or_kleene(pc.less(lat, -90), pc.greater(lat, 90)),
        pc.or_kleene(pc.less(lon, -180), pc.greater(lon, 180))), False)
    n_bad_coord = int(pc.sum(out_rng).as_py() or 0)
    coord_examples = t.filter(out_rng).select(["country", "city", "id", "latitude",
                                               "longitude"]).slice(0, 10).to_pylist()
    null_f = pa.scalar(None, pa.float64())
    t = t.set_column(t.column_names.index("latitude"), "latitude",
                     pc.if_else(out_rng, null_f, lat))
    t = t.set_column(t.column_names.index("longitude"), "longitude",
                     pc.if_else(out_rng, null_f, lon))
    info["invalid_values"] = {"coordinates_out_of_range": {
        "count": n_bad_coord, "action": "latitude and longitude set to null",
        "examples": coord_examples}}

    # 5. License category, host tags. The source license text and Airbnb's own
    #    host_listings_count are kept under *_source names.
    lic_raw = t["license"].to_pandas()
    lic_cat, lic_rule = classify_license(lic_raw)
    names = t.column_names
    t = t.set_column(names.index("license"), "license", pa.array(lic_cat, pa.string()))
    t = t.add_column(names.index("license") + 1, "license_source",
                     pa.array(lic_raw, pa.string()))
    if "host_listings_count" in t.column_names:
        t = t.rename_columns(["host_listings_count_source" if c == "host_listings_count"
                              else c for c in t.column_names])
    t = t.append_column("host_listings_count", pa.array(hcount, pa.int64(), from_pandas=True))
    t = t.append_column("host_category", pa.array(host_cat, pa.string()))

    # 6. Required fields.
    mask, by_field = required_mask(t, LISTING_REQUIRED)
    removed = int(pc.sum(mask).as_py() or 0)
    price_unparseable = by_field.get("price", 0) - price_empty
    t = t.filter(pc.invert(mask))
    info["steps"].append({
        "step": "required_field_null", "removed": removed,
        "by_field": by_field,
        "price_detail": {"empty_in_source": price_empty,
                         "unparseable": price_unparseable},
        "note": "a row missing several fields is counted once in 'removed' "
                "and once per field in 'by_field'"})

    info["rows_after"] = t.num_rows
    check_rows(info)
    info["nulls_after"] = null_counts(t)
    after_city = city_counts(t)
    info["per_city"] = per_city_table(before_city, after_city)
    info["unparseable"] = bad.report()

    # Price-empty share per city, which drives most of the row loss.
    city_drop = {k: v for k, v in info["per_city"].items() if v["removed"]}
    if removed:
        worst = sorted(city_drop.items(), key=lambda kv: -kv[1]["removed"] / kv[1]["before"])[:5]
        problems.append(
            f"listings: {removed:,} of {info['rows_before']:,} rows "
            f"({removed / info['rows_before']:.1%}) were dropped because a required field "
            f"was null; {price_empty:,} had an empty price in the source. Inside Airbnb "
            "publishes no price for these listings and there is no other price column "
            "to fall back on. Highest shares: "
            + ", ".join(f"{k} {v['removed'] / v['before']:.0%}" for k, v in worst))

    final = t.to_pandas(types_mapper={pa.int64(): pd.Int64Dtype()}.get)
    info["license_mapping"] = license_report(final["license_source"], final["license"],
                                             classify_license(final["license_source"])[1])
    info["host_category"] = {k: int((final["host_category"] == k).sum()) for k in HOST_VALUES}
    info["ranges"] = listing_ranges(final)
    problems.append(
        "listings: every price is formatted with '$' but the values are in each city's "
        "local currency (for example Budapest in HUF, Buenos Aires in ARS). The symbol is "
        "stripped; no currency conversion is done, so prices are only comparable within "
        "a city. Price ranges and the 99th percentile are therefore reported per city.")
    problems.append(
        "listings: license categories come from text rules (see license_mapping). A "
        "value with a digit counts as Licensed, but the number itself is not checked "
        "against any registry, so placeholders such as '12-3456' are counted as Licensed.")
    problems.append(
        "listings: the source already has a host_listings_count column (Airbnb's own "
        "figure). It is kept as host_listings_count_source; host_listings_count is the "
        "count of the host's listings in this data for that city. The original license "
        "text is kept as license_source.")

    atomic_write_table(t, dst)
    return info


def listing_ranges(df: pd.DataFrame) -> dict:
    out = {"price_per_city": {}, "price_above_p99_per_city": {}}
    df = df.assign(_k=df["country"] + "/" + df["city"])
    for k, g in df.groupby("_k", sort=True):
        p = g["price"]
        p99 = float(p.quantile(0.99))
        above = g[p > p99].sort_values("price", ascending=False)
        out["price_per_city"][k] = {
            "min": float(p.min()), "median": float(p.median()), "p99": p99,
            "max": float(p.max()), "zero_price": int((p == 0).sum()),
            "above_p99": int(len(above))}
        out["price_above_p99_per_city"][k] = {
            "threshold": p99, "count": int(len(above)),
            "top10": [{"id": int(r.id), "price": float(r.price)}
                      for r in above.head(10).itertuples()]}
    for col in ("minimum_nights", "number_of_reviews", "reviews_per_month",
                "latitude", "longitude"):
        s = df[col].dropna()
        out[col] = {"min": float(s.min()), "max": float(s.max()),
                    "median": float(s.median())} if len(s) else None
    mn = df["minimum_nights"]
    out["minimum_nights_over_365"] = int((mn > 365).sum())
    return out


# ---------------------------------------------------------------------------
# Calendar and reviews (streamed per city)
# ---------------------------------------------------------------------------

CALENDAR_PARSE = {"listing_id": "int", "date": "date", "available": "bool_tf",
                  "price": "price", "adjusted_price": "price",
                  "minimum_nights": "int", "maximum_nights": "int"}
CALENDAR_REQUIRED = ["listing_id", "date", "price"]
REVIEWS_PARSE = {"listing_id": "int", "id": "int", "reviewer_id": "int", "date": "date"}
REVIEWS_REQUIRED = ["listing_id", "date"]


def clean_streamed(name: str, src: Path, dst: Path, kinds: dict, required: list[str],
                   city_stats, problems: list[str]) -> dict:
    pf = pq.ParquetFile(src)
    cols = pf.schema_arrow.names
    missing = [c for c in required if c not in cols]
    if missing:
        problems.append(
            f"{name}: required column(s) {missing} do not exist in this data, so they "
            "were not used to drop rows" + (
                " and the price rules (float conversion, currency stripping) could not "
                "be applied. The Inside Airbnb calendar release used here has no price "
                "or adjusted_price columns." if "price" in missing else "."))

    bad = BadValues()
    nb, na = Counter(), Counter()
    before_city, after_city = Counter(), Counter()
    req_removed, dup_removed, by_field = 0, 0, Counter()
    stats: dict = {}
    fffd = Counter()
    tmp = dst.with_name(dst.name + ".tmp")
    writer = None
    try:
        for country, city, groups in city_row_groups(src):
            label = city_label(country, city)
            t = pf.read_row_groups(groups)
            n0 = t.num_rows
            before_city[label] = n0
            nb.update(raw_null_counts(t))
            for c in ("comments", "reviewer_name"):
                if c in cols:
                    fffd[c] += int(pc.sum(pc.match_substring(t[c], "\ufffd")).as_py() or 0)
            t = pa.table({c: empty_to_null(t[c]) for c in t.column_names})
            t = apply_parsers(t, kinds, bad)

            mask, bf = required_mask(t, required)
            by_field.update(bf)
            t = t.filter(pc.invert(mask))
            n1 = t.num_rows
            req_removed += n0 - n1

            t = t.group_by(t.column_names, use_threads=False).aggregate([])
            dup_removed += n1 - t.num_rows

            if writer is None:
                schema = t.schema
                writer = pq.ParquetWriter(tmp, schema)
            t = t.select(schema.names).cast(schema)
            writer.write_table(t)
            after_city[label] = t.num_rows
            na.update(null_counts(t))
            stats[label] = city_stats(t)
            print(f"{name:<10} {label:<36} {n0:>12,} -> {t.num_rows:>12,}")
            del t
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, dst)

    info = {
        "rows_before": sum(before_city.values()),
        "steps": [
            {"step": "required_field_null", "removed": req_removed,
             "by_field": dict(by_field),
             "required_fields_present": [c for c in required if c in cols],
             "note": "a row missing several fields is counted once in 'removed' "
                     "and once per field in 'by_field'"},
            {"step": "exact_duplicate_rows", "removed": dup_removed},
        ],
        "rows_after": sum(after_city.values()),
        "nulls_before": {c: nb[c] for c in cols},
        "nulls_after": {c: na[c] for c in cols},
        "per_city": per_city_table(before_city, after_city),
        "unparseable": bad.report(),
        "ranges": {"per_city": stats},
    }
    check_rows(info)
    fffd = {c: n for c, n in fffd.items() if n}
    if fffd:
        problems.append(
            f"{name}: some text contains the Unicode replacement character (U+FFFD), "
            "meaning characters were already lost in the source files. Rows affected "
            f"per column: {fffd}")
    return info


def _date_range(t: pa.Table) -> dict:
    mm = pc.min_max(t["date"]).as_py()
    return {"date_min": str(mm["min"]) if mm["min"] else None,
            "date_max": str(mm["max"]) if mm["max"] else None}


def calendar_stats(t: pa.Table) -> dict:
    out = _date_range(t)
    out["rows"] = t.num_rows
    out["available_share"] = round(float(pc.mean(pc.cast(t["available"], pa.int8())).as_py() or 0), 4)
    if "minimum_nights" in t.column_names:
        mm = pc.min_max(t["minimum_nights"]).as_py()
        out["minimum_nights_min"], out["minimum_nights_max"] = mm["min"], mm["max"]
    for c in ("price", "adjusted_price"):
        if c in t.column_names:
            mm = pc.min_max(t[c]).as_py()
            out[f"{c}_min"], out[f"{c}_max"] = mm["min"], mm["max"]
    return out


def reviews_stats(t: pa.Table) -> dict:
    out = _date_range(t)
    out["rows"] = t.num_rows
    out["listings_with_reviews"] = int(pc.count_distinct(t["listing_id"]).as_py())
    return out


# ---------------------------------------------------------------------------
# Neighbourhoods
# ---------------------------------------------------------------------------

def clean_neighbourhoods(src: Path, dst: Path) -> dict:
    gdf = gpd.read_parquet(src)
    geom = gdf.geometry.name

    def nulls(g):
        return {c: int(g[c].isna().sum() + ((g[c] == "").sum() if g[c].dtype == object
                                            and c != geom else 0))
                for c in g.columns}

    def per_city(g):
        return Counter((g["country"] + "/" + g["city"]).value_counts().to_dict())

    info = {"rows_before": len(gdf), "nulls_before": nulls(gdf)}
    before_city = per_city(gdf)
    bad = gdf.geometry.isna() | gdf.geometry.is_empty
    examples = gdf.loc[bad, ["country", "city"] + [c for c in gdf.columns
                                                   if c not in (geom, "country", "city")]]
    gdf = gdf[~bad]
    info["steps"] = [{"step": "null_or_empty_geometry", "removed": int(bad.sum()),
                      "examples": examples.head(10).astype(str).to_dict("records")}]
    info["rows_after"] = len(gdf)
    check_rows(info)
    info["nulls_after"] = nulls(gdf)
    info["per_city"] = per_city_table(before_city, per_city(gdf))
    info["unparseable"] = {}
    info["crs"] = str(gdf.crs)
    tmp = dst.with_name(dst.name + ".tmp")
    gdf.to_parquet(tmp, index=False)
    os.replace(tmp, dst)
    return info


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return "" if v is None else str(v)


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    def cell(v):
        return fmt(v).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return out + [""]


def render_markdown(report: dict) -> str:
    L = ["# TourisStay cleaning report", ""]
    L += ["## Summary", ""]
    L += md_table(["Table", "Rows before", "Removed", "Rows after", "Adds up"],
                  [[n, i["rows_before"], i["rows_before"] - i["rows_after"],
                    i["rows_after"], i["rows_add_up"]]
                   for n, i in report["tables"].items()])

    L += ["## Problems not fixed", ""]
    L += [f"- {p}" for p in report["problems"]] or ["- None"]
    L += [""]

    for name, info in report["tables"].items():
        L += [f"## {name}", ""]
        L += ["### Rows removed by step", ""]
        rows = [["(start)", "", info["rows_before"]]]
        left = info["rows_before"]
        for s in info["steps"]:
            left -= s["removed"]
            rows.append([s["step"], s["removed"], left])
        L += md_table(["Step", "Removed", "Rows left"], rows)
        for s in info["steps"]:
            if s.get("by_field"):
                L += [f"Nulls by required field (`{s['step']}`): "
                      + ", ".join(f"{k} {fmt(v)}" for k, v in s["by_field"].items())]
                if "price_detail" in s:
                    d = s["price_detail"]
                    L += [f"Price: {fmt(d['empty_in_source'])} empty in source, "
                          f"{fmt(d['unparseable'])} unparseable."]
                L += [""]

        if info.get("invalid_values"):
            for k, v in info["invalid_values"].items():
                L += [f"Invalid values, {k}: {fmt(v['count'])} ({v['action']})", ""]

        L += ["### Values that could not be parsed", ""]
        if info["unparseable"]:
            for col, u in info["unparseable"].items():
                ex = ", ".join(f"`{e['value']}` ({fmt(e['count'])})" for e in u["examples"])
                L += [f"- **{col}**: {fmt(u['count'])} values ({fmt(u['distinct'])} distinct). "
                      f"Examples: {ex}"]
        else:
            L += ["- None"]
        L += [""]

        if "license_mapping" in info:
            lm = info["license_mapping"]
            L += ["### License mapping", ""]
            L += md_table(["Category", "Rows"], [[k, v] for k, v in lm["categories"].items()])
            L += md_table(["Category", "Rule", "Rows"],
                          [[r["category"], r["rule"], r["rows"]] for r in lm["rules"]])
            for cat, d in lm["by_category"].items():
                if not d["rows"]:
                    continue
                L += [f"**{cat}**: {fmt(d['rows'])} rows, {fmt(d['distinct_raw_values'])} "
                      "distinct raw values.", ""]
                if d["text_values"]:
                    L += ["Top raw values without a digit:", ""]
                    L += md_table(["Raw value", "Count"],
                                  [[e["value"] or "(empty)", e["count"]]
                                   for e in d["text_values"][:15]])
                if d["numbered_value_shapes_top50"]:
                    L += ["Top numbered value shapes (digit = 9, letter = A):", ""]
                    L += md_table(["Shape", "Count", "Example"],
                                  [[e["shape"], e["count"], e["example"]]
                                   for e in d["numbered_value_shapes_top50"][:10]])
            L += ["The full list of text values and the top 50 shapes per category are in "
                  "cleaning_report.json.", ""]
        if "host_category" in info:
            L += ["### Host category", ""]
            L += md_table(["Category", "Rows"], [[k, v] for k, v in info["host_category"].items()])

        r = info.get("ranges", {})
        if "price_per_city" in r:
            L += ["### Price per city (local currency)", ""]
            L += md_table(["City", "Min", "Median", "P99", "Max", "Zero price", "Above P99"],
                          [[k, v["min"], v["median"], v["p99"], v["max"], v["zero_price"],
                            v["above_p99"]] for k, v in r["price_per_city"].items()])
            other = [[c, r[c]["min"], r[c]["median"], r[c]["max"]]
                     for c in ("minimum_nights", "number_of_reviews", "reviews_per_month",
                               "latitude", "longitude") if r.get(c)]
            L += ["### Other ranges", ""]
            L += md_table(["Column", "Min", "Median", "Max"], other)
            L += [f"Listings with minimum_nights over 365: "
                  f"{fmt(r['minimum_nights_over_365'])}", ""]
        elif "per_city" in r:
            L += ["### Ranges per city", ""]
            keys = [k for k in next(iter(r["per_city"].values()), {}) if k != "rows"]
            L += md_table(["City"] + keys,
                          [[c] + [v.get(k) for k in keys] for c, v in r["per_city"].items()])

        L += ["### Rows per city", ""]
        L += md_table(["City", "Before", "After", "Removed"],
                      [[k, v["before"], v["after"], v["removed"]]
                       for k, v in info["per_city"].items()])

        L += ["### Nulls per column", ""]
        cols = list(dict.fromkeys(list(info["nulls_before"]) + list(info["nulls_after"])))
        L += md_table(["Column", "Before", "After"],
                      [[c, info["nulls_before"].get(c, "-"), info["nulls_after"].get(c, "-")]
                       for c in cols])
    return "\n".join(L)


# ---------------------------------------------------------------------------

def main() -> int:
    # The report contains non-ASCII text; never fail on a narrow console encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = data_dir()
    src, dst = base / "integrated", base / "cleaned"
    needed = [src / f"{t}.parquet" for t in ("listings", "calendar", "reviews", "neighbourhoods")]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        print(f"ERROR missing input(s): {[str(p) for p in missing]}; run ingest.py first",
              file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    tables: dict[str, dict] = {}
    print("Cleaning listings")
    tables["listings"] = clean_listings(src / "listings.parquet", dst / "listings.parquet",
                                        problems)
    print("Cleaning calendar")
    tables["calendar"] = clean_streamed("calendar", src / "calendar.parquet",
                                        dst / "calendar.parquet", CALENDAR_PARSE,
                                        CALENDAR_REQUIRED, calendar_stats, problems)
    print("Cleaning reviews")
    tables["reviews"] = clean_streamed("reviews", src / "reviews.parquet",
                                       dst / "reviews.parquet", REVIEWS_PARSE,
                                       REVIEWS_REQUIRED, reviews_stats, problems)
    print("Cleaning neighbourhoods")
    tables["neighbourhoods"] = clean_neighbourhoods(src / "neighbourhoods.parquet",
                                                    dst / "neighbourhoods.parquet")

    for name, info in tables.items():
        for col, u in info["unparseable"].items():
            problems.append(f"{name}: {u['count']:,} value(s) in '{col}' could not be "
                            "parsed and were set to null (examples in the report).")

    report = {"tables": tables, "problems": problems}
    with open(dst / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    md = render_markdown(report)
    (dst / "cleaning_report.md").write_text(md + "\n", encoding="utf-8")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
