"""Download monthly rainfall from NASA POWER for the Mini-Project 4 regions.

Source: NASA Langley Research Center POWER Project, https://power.larc.nasa.gov/
Parameter: PRECTOTCORR_SUM (bias-corrected total precipitation, mm per month),
community AG, monthly temporal API.

Usage (from the repository root)::

    python scripts/fetch_nasa_power.py

Writes ``data/nasa_power_monthly_rainfall.csv`` with columns
region, lat, lon, year, month, rainfall_mm. The notebook reads that CSV, so it
runs offline; this script only needs to be run again to refresh the data.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

REGIONS = {
    "Kampala": (0.3476, 32.5825),
    "Gulu": (2.7724, 32.2881),
    "Mbarara": (-0.6072, 30.6545),
}
START, END = 2014, 2023
URL = (
    "https://power.larc.nasa.gov/api/temporal/monthly/point"
    "?parameters=PRECTOTCORR_SUM&community=AG&longitude={lon}&latitude={lat}"
    "&start={start}&end={end}&format=JSON"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "nasa_power_monthly_rainfall.csv"


def fetch(lat: float, lon: float) -> dict[str, float]:
    """Return the {YYYYMM: mm} mapping for one point (month '13' = annual total)."""
    url = URL.format(lat=lat, lon=lon, start=START, end=END)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.load(resp)
    return payload["properties"]["parameter"]["PRECTOTCORR_SUM"]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["region", "lat", "lon", "year", "month", "rainfall_mm"])
        for region, (lat, lon) in REGIONS.items():
            series = fetch(lat, lon)
            for key, value in sorted(series.items()):
                year, month = int(key[:4]), int(key[4:])
                if month == 13 or value < 0:  # skip annual totals and fill values (-999)
                    continue
                writer.writerow([region, lat, lon, year, month, f"{value:.2f}"])
            print(f"{region}: {len(series)} values")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
