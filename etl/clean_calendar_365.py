"""Check and fix the 365-day forward calendar for each listing.

Reads   <DATA_DIR>/cleaned/calendar.parquet          (from clean.py)
Writes  <DATA_DIR>/cleaned/calendar_365.parquet
        <DATA_DIR>/cleaned/calendar_365_report.json and calendar_365_report.md

One output row per (country, city, listing_id, date), with flag columns.
Rows are flagged, not dropped; the only rows removed are duplicate keys, and
the only rows added are gap rows with no availability and no price. No
aggregation happens here.

The window per city
    Inside Airbnb scrapes a city over several days, and each listing's
    calendar runs 365 days from the day that listing was scraped. The
    expected window for a city therefore runs from the first scrape day (the
    earliest first date of any listing) to 364 days after the last scrape day
    (the latest first date of any listing). Rows outside it are flagged
    out_of_window. The raw earliest-to-latest span is reported too.

Prices
    If the input has no price / adjusted_price column (the Inside Airbnb
    release used for this project has neither), the columns are added as
    all-null floats so the schema is stable, and the price-based flags are
    left null: they could not be evaluated, which is different from false.

The calendar is processed one city at a time (the largest city has ~34M rows).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from clean import city_row_groups, fmt, md_table
from ingest import data_dir

DAYS = 365
WINDOW_TOLERANCE_DAYS = 30      # raw span this far from 365 is "much" off
SCRAPE_SPREAD_LIMIT_DAYS = 14   # scrape days spread wider than this is suspicious
KEY = ["listing_id", "date"]

FLAGS = ["out_of_window", "incomplete_calendar", "is_gap_filled",
         "price_missing_when_available", "price_invalid", "price_outlier",
         "price_adjustment_large", "unavailable_with_price", "min_nights_applies"]
PRICE_FLAGS = ["price_missing_when_available", "price_invalid", "price_outlier",
               "price_adjustment_large", "unavailable_with_price"]

NULLABLE = {pa.int64(): pd.Int64Dtype(), pa.bool_(): pd.BooleanDtype(),
            pa.float64(): pd.Float64Dtype()}


def iso(ts) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).date().isoformat()


def num(v):
    return None if v is None or pd.isna(v) else float(v)


def fill_gaps(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Rows for every missing date between a listing's first and last date."""
    gl = stats[stats["missing_between"] > 0]
    if gl.empty:
        return df.iloc[0:0]
    reps = gl["span"].to_numpy()
    ids = np.repeat(gl.index.to_numpy(), reps)
    offsets = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    dates = np.repeat(gl["first"].to_numpy(), reps) + offsets.astype("timedelta64[D]")
    full = pd.DataFrame({"listing_id": pd.array(ids, dtype="Int64"),
                         "date": pd.to_datetime(dates)})
    have = df.loc[df["listing_id"].isin(gl.index), KEY]
    full = full.merge(have, on=KEY, how="left", indicator=True)
    full = full.loc[full["_merge"] == "left_only", KEY].reset_index(drop=True)
    gaps = pd.DataFrame({c: pd.Series(pd.NA, index=full.index, dtype=df[c].dtype)
                         for c in df.columns if c not in KEY})
    gaps[KEY] = full[KEY]
    return gaps[df.columns]


def clean_city(df: pd.DataFrame, has_price: bool) -> tuple[pd.DataFrame, dict]:
    info: dict = {"rows_before": len(df)}

    # Duplicate keys: keep the row with a non-null price, else the first one.
    df = df.assign(_nullprice=df["price"].isna() if has_price else False)
    df = df.sort_values(KEY + ["_nullprice"], kind="mergesort")
    dup = df.duplicated(KEY, keep="first")
    info["duplicates_removed"] = int(dup.sum())
    df = df.loc[~dup].drop(columns="_nullprice")

    # Per-listing coverage, from the real (non-gap) rows.
    g = df.groupby("listing_id", sort=False)["date"]
    stats = pd.DataFrame({"first": g.min(), "last": g.max(), "n_dates": g.size()})
    stats["span"] = (stats["last"] - stats["first"]).dt.days + 1
    stats["missing_between"] = stats["span"] - stats["n_dates"]
    stats["missing_vs_365"] = (DAYS - stats["n_dates"]).clip(lower=0)

    # Window.
    raw_min, raw_max = df["date"].min(), df["date"].max()
    start, last_scrape = stats["first"].min(), stats["first"].max()
    end = last_scrape + pd.Timedelta(days=DAYS - 1)
    raw_len = int((raw_max - raw_min).days + 1)
    spread = int((last_scrape - start).days)
    info["window"] = {
        "earliest_date": iso(raw_min), "latest_date": iso(raw_max), "raw_length_days": raw_len,
        "first_scrape_day": iso(start), "last_scrape_day": iso(last_scrape),
        "scrape_spread_days": spread,
        "window_start": iso(start), "window_end": iso(end),
        "window_length_days": int((end - start).days + 1),
        "flagged": abs(raw_len - DAYS) > WINDOW_TOLERANCE_DAYS
                   or spread > SCRAPE_SPREAD_LIMIT_DAYS,
    }

    # Gap rows, only between a listing's first and last date inside the
    # window, so a stray out-of-window date does not create years of rows.
    inw = df.loc[(df["date"] >= start) & (df["date"] <= end)]
    gw = inw.groupby("listing_id", sort=False)["date"]
    gstats = pd.DataFrame({"first": gw.min(), "last": gw.max(), "n_dates": gw.size()})
    gstats["span"] = (gstats["last"] - gstats["first"]).dt.days + 1
    gstats["missing_between"] = gstats["span"] - gstats["n_dates"]
    gaps = fill_gaps(df, gstats)
    info["gap_rows_created"] = len(gaps)
    df = df.assign(is_gap_filled=False)
    if len(gaps):
        df = pd.concat([df, gaps.assign(is_gap_filled=True)], ignore_index=True)
        df = df.sort_values(KEY, kind="mergesort", ignore_index=True)

    df["out_of_window"] = (df["date"] < start) | (df["date"] > end)
    df["incomplete_calendar"] = df["listing_id"].map(stats["n_dates"] < DAYS).astype(bool)
    inc = stats[stats["n_dates"] < DAYS]
    info["incomplete_listings"] = int(len(inc))
    info["listings"] = int(len(stats))
    info["median_missing_days_incomplete"] = num(inc["missing_vs_365"].median()) if len(inc) else None
    info["listings_over_365_dates"] = int((stats["n_dates"] > DAYS).sum())

    avail = df["available"]
    if has_price:
        bad = (df["price"] <= 0).fillna(False)
        info["prices_set_to_null"] = int(bad.sum())
        df["price_invalid"] = bad
        df.loc[bad, "price"] = pd.NA
        price = df["price"]
        df["price_missing_when_available"] = (avail.fillna(False) & price.isna()).astype(bool)
        valid = price.dropna()
        p1, p99 = (num(valid.quantile(0.01)), num(valid.quantile(0.99))) if len(valid) else (None, None)
        df["price_outlier"] = ((price < p1) | (price > p99)).fillna(False).astype(bool) \
            if p1 is not None else False
        adj = df["adjusted_price"]
        df["price_adjustment_large"] = ((adj - price).abs() / price > 0.5).fillna(False).astype(bool)
        df["unavailable_with_price"] = ((avail == False) & price.notna()).fillna(False).astype(bool)  # noqa: E712
        info["price_stats"] = {"min": num(valid.min()), "median": num(valid.median()),
                               "mean": num(valid.mean()), "max": num(valid.max()),
                               "p1": p1, "p99": p99, "rows_with_price": int(len(valid))}
    else:
        for f in PRICE_FLAGS:
            df[f] = pd.array([pd.NA] * len(df), dtype="boolean")
        info["prices_set_to_null"] = None
        info["price_stats"] = None

    df["min_nights_applies"] = ((df["minimum_nights"] > 1) & avail).fillna(False).astype(bool)

    info["flags"] = {f: (None if df[f].isna().all() else int(df[f].fillna(False).sum()))
                     for f in FLAGS}

    # Share of available days per month (real rows only).
    real = df.loc[~df["is_gap_filled"] & avail.notna(), ["date", "available"]]
    month = real["date"].dt.strftime("%Y-%m")
    share = real["available"].astype(float).groupby(month).agg(["mean", "size"])
    info["available_share_by_month"] = {m: {"share": round(float(r["mean"]), 4), "days": int(r["size"])}
                                        for m, r in share.iterrows()}

    info["rows_after"] = len(df)
    info["rows_add_up"] = (info["rows_before"] + info["gap_rows_created"]
                           - info["duplicates_removed"] == info["rows_after"])
    if not info["rows_add_up"]:
        raise RuntimeError(f"row arithmetic does not add up: {info}")
    return df, info


def render_markdown(report: dict) -> str:
    cities, tot = report["cities"], report["totals"]
    L = ["# TourisStay calendar 365 report", ""]
    L += ["## Totals", ""]
    L += md_table(["Rows before", "Gap rows created", "Duplicates removed", "Rows after", "Adds up"],
                  [[tot["rows_before"], tot["gap_rows_created"], tot["duplicates_removed"],
                    tot["rows_after"], tot["rows_add_up"]]])
    L += md_table(["Flag", "Rows"], [[f, "not evaluated" if v is None else v]
                                     for f, v in tot["flags"].items()])
    L += ["## Problems not fixed", ""]
    L += [f"- {p}" for p in report["problems"]] or ["- None"]
    L += [""]

    L += ["## Window per city", ""]
    L += md_table(["City", "Earliest", "Latest", "Raw days", "Scrape days", "Window start",
                   "Window end", "Window days", "Flagged"],
                  [[c, w["earliest_date"], w["latest_date"], w["raw_length_days"],
                    w["scrape_spread_days"] + 1, w["window_start"], w["window_end"],
                    w["window_length_days"], "YES" if w["flagged"] else ""]
                   for c, i in cities.items() for w in [i["window"]]])

    L += ["## Rows per city", ""]
    L += md_table(["City", "Before", "Gap rows", "Duplicates", "After", "Listings",
                   "Incomplete", "Median missing days", "Over 365 dates"],
                  [[c, i["rows_before"], i["gap_rows_created"], i["duplicates_removed"],
                    i["rows_after"], i["listings"], i["incomplete_listings"],
                    i["median_missing_days_incomplete"], i["listings_over_365_dates"]]
                   for c, i in cities.items()])

    L += ["## Flags per city", "", "Blank cells in price flags mean not evaluated (no price data).", ""]
    L += md_table(["City"] + FLAGS,
                  [[c] + [i["flags"][f] for f in FLAGS] for c, i in cities.items()])

    L += ["## Price statistics per city (after cleaning, local currency)", ""]
    if any(i["price_stats"] for i in cities.values()):
        L += md_table(["City", "Min", "P1", "Median", "Mean", "P99", "Max", "Rows with price"],
                      [[c] + ([s[k] for k in ("min", "p1", "median", "mean", "p99", "max",
                                              "rows_with_price")] if s else ["n/a"] * 7)
                       for c, i in cities.items() for s in [i["price_stats"]]])
    else:
        L += ["No price data in the input calendar.", ""]

    L += ["## Share of available days per month", "",
          "First and last months are partial (the window starts and ends mid-month).", ""]
    months = sorted({m for i in cities.values() for m in i["available_share_by_month"]})
    L += md_table(["City"] + months,
                  [[c] + [f"{i['available_share_by_month'][m]['share']:.0%}"
                          if m in i["available_share_by_month"] else "" for m in months]
                   for c, i in cities.items()])
    return "\n".join(L)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    base = data_dir() / "cleaned"
    src = base / "calendar.parquet"
    dst = base / "calendar_365.parquet"
    if not src.is_file():
        print(f"ERROR missing input {src}; run clean.py first", file=sys.stderr)
        return 1

    pf = pq.ParquetFile(src)
    in_cols = pf.schema_arrow.names
    data_cols = [c for c in in_cols if c not in ("country", "city")]
    has_price = "price" in in_cols
    problems: list[str] = []
    if not has_price:
        problems.append(
            "The input calendar has no price column (the Inside Airbnb release used here "
            "publishes none). price and adjusted_price are added as all-null floats, and the "
            "price flags (price_missing_when_available, price_invalid, price_outlier, "
            "price_adjustment_large, unavailable_with_price) are null: not evaluated. Price "
            "statistics cannot be reported.")
    elif "adjusted_price" not in in_cols:
        problems.append("The input calendar has no adjusted_price column; it is added as "
                        "all-null and price_adjustment_large is never true.")

    out_cols = data_cols + [c for c in ("price", "adjusted_price") if c not in data_cols]
    out_cols += ["country", "city"] + FLAGS
    cities: dict[str, dict] = {}
    tmp = dst.with_name(dst.name + ".tmp")
    writer, schema = None, None
    try:
        for country, city, groups in city_row_groups(src):
            label = f"{country}/{city}"
            df = pf.read_row_groups(groups, columns=data_cols).to_pandas(
                date_as_object=False, types_mapper=NULLABLE.get)
            for c in ("price", "adjusted_price"):
                if c not in df.columns:
                    df[c] = pd.array([pd.NA] * len(df), dtype="Float64")
            df, info = clean_city(df, has_price)
            cities[label] = info

            t = pa.Table.from_pandas(df, preserve_index=False)
            n = t.num_rows
            t = t.append_column("country", pa.array([country] * n, pa.string()))
            t = t.append_column("city", pa.array([city] * n, pa.string()))
            t = t.select(out_cols)
            if schema is None:
                fields = []
                for f in t.schema:
                    typ = {"date": pa.date32(), "price": pa.float64(),
                           "adjusted_price": pa.float64()}.get(f.name, f.type)
                    if f.name in FLAGS or f.name == "available":
                        typ = pa.bool_()
                    fields.append(pa.field(f.name, typ))
                schema = pa.schema(fields)
                writer = pq.ParquetWriter(tmp, schema)
            writer.write_table(t.cast(schema))
            print(f"calendar_365 {label:<36} {info['rows_before']:>12,} -> {info['rows_after']:>12,}"
                  f"  gaps +{info['gap_rows_created']:,}  dups -{info['duplicates_removed']:,}")
            del df, t
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, dst)

    totals = {k: sum(i[k] for i in cities.values())
              for k in ("rows_before", "gap_rows_created", "duplicates_removed", "rows_after",
                        "listings", "incomplete_listings")}
    totals["rows_add_up"] = (totals["rows_before"] + totals["gap_rows_created"]
                             - totals["duplicates_removed"] == totals["rows_after"]
                             == pq.ParquetFile(dst).metadata.num_rows)
    flag_tot: Counter = Counter()
    for i in cities.values():
        for f, v in i["flags"].items():
            if v is not None:
                flag_tot[f] += v
    totals["flags"] = {f: (flag_tot[f] if f in flag_tot else None) for f in FLAGS}

    flagged = [c for c, i in cities.items() if i["window"]["flagged"]]
    if flagged:
        problems.append(
            f"Cities whose window is much shorter or longer than {DAYS} days (more than "
            f"{WINDOW_TOLERANCE_DAYS} days off) or whose scrape days spread over more than "
            f"{SCRAPE_SPREAD_LIMIT_DAYS} days; their rows are kept: {', '.join(flagged)}")
    oow = totals["flags"]["out_of_window"]
    if oow:
        problems.append(f"{oow:,} rows fall outside their city's window. They are kept with "
                        "out_of_window = true for you to decide.")
    if totals["incomplete_listings"]:
        problems.append(f"{totals['incomplete_listings']:,} listings have fewer than {DAYS} "
                        "dates (incomplete_calendar = true). Gaps inside a listing's range are "
                        "filled with empty rows; missing days before its first or after its "
                        "last date are not created.")
    if totals["gap_rows_created"]:
        problems.append("Gap rows have no availability, price, minimum_nights or "
                        "maximum_nights; nothing is invented for them.")
    over = sum(i["listings_over_365_dates"] for i in cities.values())
    if over:
        problems.append(f"{over:,} listings have more than {DAYS} dates. Their calendar "
                        "runs longer than one year; they are kept as they are.")

    report = {"cities": cities, "totals": totals, "problems": problems,
              "window_rule": "window_start = earliest first date of any listing in the city "
                             "(first scrape day); window_end = latest first date + 364 days"}
    with open(base / "calendar_365_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    md = render_markdown(report)
    (base / "calendar_365_report.md").write_text(md + "\n", encoding="utf-8")
    print()
    print(md)
    return 0 if totals["rows_add_up"] else 1


if __name__ == "__main__":
    sys.exit(main())
