"""
fetch_wind_data.py

Downloads historical wind speed and wind gust data for two contrasting
sites via the free Open-Meteo Historical Weather API (ERA5/ERA5-Land
reanalysis, back to 1940, no API key required, free for non-commercial
use). Prepares the two series needed for the wind case study (Section:
Case Study - Wind Speed Modeling and Extreme Wind Gusts):

    1. Hourly wind speed  -> Weibull fit (bulk distribution)
    2. Monthly maximum wind gust -> GEV/Gumbel fit (Block Maxima)

Run locally (not in a sandboxed/restricted-network environment):

    pip install requests pandas
    python fetch_wind_data.py

Outputs, per site, in the current directory:
    wind_<site>_hourly_raw.csv        -- full hourly windspeed_10m, windgusts_10m
    wind_<site>_monthly_gust_max.csv  -- monthly block maxima of daily max gust

Default sites (edit SITES below to change):
    "gusty_plains"   -- Amarillo, TX, USA      (expected lower Weibull k, ~1.5-2.0)
    "steady_coastal" -- Honolulu, HI, USA       (steady trade winds, expected higher k, ~2.5-3.5)

Data source: Open-Meteo Historical Weather API, ERA5 reanalysis.
https://open-meteo.com/en/docs/historical-weather-api
Attribution required under CC BY 4.0 (Open-Meteo, ERA5/Copernicus).
"""

import time
import requests
import pandas as pd

# ------------------------------------------------------------------
# 1. Define sites: (name, latitude, longitude)
#    Pick sites with contrasting wind regimes, per the case study's
#    cross-site comparison design. Edit freely.
# ------------------------------------------------------------------
SITES = {
    "gusty_plains":   (35.2220, -101.8313),  # Amarillo, TX, USA
    "steady_coastal": (21.3069, -157.8583),  # Honolulu, HI, USA
}

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"   # adjust to today's date if you want the freshest data

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_site(name: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch hourly wind speed and daily max gust for one site."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "windspeed_10m",
        "daily": "windgusts_10m_max,windspeed_10m_max",
        "windspeed_unit": "ms",   # meters/second, standard for wind-energy analysis
        "timezone": "UTC",
    }
    print(f"Requesting {name} ({lat}, {lon}) from Open-Meteo...")
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = pd.DataFrame({
        "time": data["hourly"]["time"],
        "windspeed_10m": data["hourly"]["windspeed_10m"],
    })
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly = hourly.dropna()

    daily = pd.DataFrame({
        "date": data["daily"]["time"],
        "windgusts_10m_max": data["daily"]["windgusts_10m_max"],
        "windspeed_10m_max": data["daily"]["windspeed_10m_max"],
    })
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.dropna()

    return hourly, daily


def main():
    for site_name, (lat, lon) in SITES.items():
        hourly, daily = fetch_site(site_name, lat, lon)

        # --- Save full hourly wind speed series (for Weibull fit) ---
        hourly_path = f"wind_{site_name}_hourly_raw.csv"
        hourly.to_csv(hourly_path, index=False)
        print(f"  Saved {len(hourly)} hourly rows to {hourly_path}")

        # --- Build monthly block maxima of daily max gust (for GEV fit) ---
        daily["year_month"] = daily["date"].dt.to_period("M")
        monthly_max = (
            daily.groupby("year_month")["windgusts_10m_max"]
            .max()
            .reset_index()
            .rename(columns={"windgusts_10m_max": "monthly_max_gust_ms"})
        )
        monthly_path = f"wind_{site_name}_monthly_gust_max.csv"
        monthly_max.to_csv(monthly_path, index=False)
        print(f"  Saved {len(monthly_max)} monthly block maxima to {monthly_path}")

        # --- Quick diagnostics ---
        n_years = hourly["time"].dt.year.nunique()
        mean_speed = hourly["windspeed_10m"].mean()
        print(f"  {site_name}: ~{n_years} years, mean wind speed = {mean_speed:.2f} m/s, "
              f"{len(monthly_max)} monthly blocks for GEV fit\n")

        time.sleep(1)  # be polite to the free API


if __name__ == "__main__":
    main()
