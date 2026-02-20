import os
import re
import math
import gzip
import io
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import requests
import streamlit as st
from geopy.geocoders import Nominatim

# -----------------------------
# NOAA / NCEI endpoints (ISD)
# -----------------------------
NCEI_NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/noaa"
ISD_HISTORY_URL = f"{NCEI_NOAA_BASE}/isd-history.csv"
ISD_INVENTORY_URL = f"{NCEI_NOAA_BASE}/isd-inventory.csv"
COUNTRY_LIST_URL = f"{NCEI_NOAA_BASE}/country-list.txt"

# ISD-Lite files are at:
# https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/YYYY/USAF-WBAN-YYYY.gz
ISD_LITE_BASE = f"{NCEI_NOAA_BASE}/isd-lite"

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache_isd")
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def safe_read_csv(url: str, cache_name: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_station_history() -> pd.DataFrame:
    df = safe_read_csv(ISD_HISTORY_URL, "isd-history.csv")
    # Standardize column names just in case
    df.columns = [c.strip().upper() for c in df.columns]
    # Keep only rows with coords
    df = df.dropna(subset=["LAT", "LON"])
    return df

@st.cache_data(show_spinner=False)
def load_inventory() -> pd.DataFrame:
    df = safe_read_csv(ISD_INVENTORY_URL, "isd-inventory.csv")
    df.columns = [c.strip().upper() for c in df.columns]
    # Expected: USAF, WBAN, YEAR, ... (other fields may exist)
    return df

@st.cache_data(show_spinner=False)
def load_country_list() -> Dict[str, str]:
    """
    country-list.txt is FIPS-ish 2-letter codes + names in one long line.
    We'll parse pairs like: "EI IRELAND"
    """
    r = requests.get(COUNTRY_LIST_URL, timeout=60)
    r.raise_for_status()
    text = r.text.replace("\n", " ")
    # Two-letter code then country name in caps until next two-letter code
    # This file is quirky; we parse by scanning tokens.
    tokens = text.split()
    # First tokens are headers: "FIPS", "ID", "COUNTRY", "NAME"
    # Then: CODE NAME... CODE NAME...
    # We'll treat token[4:] as alternating blocks starting with 2-char code.
    out = {}
    i = 0
    # Find first 2-letter token after headers
    while i < len(tokens) and not (len(tokens[i]) == 2 and tokens[i].isalpha()):
        i += 1
    while i < len(tokens):
        code = tokens[i]
        if not (len(code) == 2 and code.isalpha()):
            i += 1
            continue
        i += 1
        name_parts = []
        while i < len(tokens) and not (len(tokens[i]) == 2 and tokens[i].isalpha()):
            name_parts.append(tokens[i])
            i += 1
        name = " ".join(name_parts).strip()
        if name:
            out[name.lower()] = code.upper()
    return out

def geocode_location(q: str) -> Optional[Tuple[float, float, str]]:
    geolocator = Nominatim(user_agent="isd_weekly_climatology_app")
    loc = geolocator.geocode(q, exactly_one=True, addressdetails=False, timeout=30)
    if not loc:
        return None
    return (float(loc.latitude), float(loc.longitude), str(loc.address))

def station_id(usaf: int, wban: int) -> str:
    return f"{int(usaf):06d}-{int(wban):05d}"

def isd_lite_url(usaf: int, wban: int, year: int) -> str:
    return f"{ISD_LITE_BASE}/{year}/{station_id(usaf,wban)}-{year}.gz"

def local_isd_lite_path(usaf: int, wban: int, year: int) -> str:
    return os.path.join(CACHE_DIR, "isd-lite", str(year), f"{station_id(usaf,wban)}-{year}.gz")

def download_isd_lite(usaf: int, wban: int, year: int) -> Optional[str]:
    path = local_isd_lite_path(usaf, wban, year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = isd_lite_url(usaf, wban, year)
    r = requests.get(url, stream=True, timeout=90)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    return path

def parse_isd_lite_gz(path: str) -> pd.DataFrame:
    """
    ISD-Lite fixed-width fields (per NCEI docs):
    year(4) month(2) day(2) hour(2) temp(6) dew(6) slp(6) wd(6) ws(6) sky(6) p1(6) p6(6)
    Values scaled by 10 for temp/dew/slp/ws/p1/p6; missing = -9999; trace precip = -1
    """
    # read whole file
    with gzip.open(path, "rb") as f:
        raw = f.read()
    # ISD-Lite is space-separated fixed width with some spaces; easiest is read_fwf
    data = pd.read_fwf(
        io.BytesIO(raw),
        widths=[4,2,2,2,6,6,6,6,6,6,6,6],
        header=None,
        names=["year","month","day","hour","temp","dew","slp","wdir","wspd","sky","p1","p6"],
    )
    # build datetime
    data["dt"] = pd.to_datetime(
        data[["year","month","day","hour"]].astype(int),
        errors="coerce",
        utc=True
    )
    data = data.dropna(subset=["dt"])
    # scale / clean
    def scale(col, factor):
        x = data[col].astype(float)
        x = x.where(x != -9999, np.nan)
        return x / factor

    data["temp_c"] = scale("temp", 10.0)
    data["dew_c"]  = scale("dew", 10.0)
    data["slp_hpa"]= scale("slp", 10.0)
    data["wspd_ms"]= scale("wspd", 10.0)
    data["sky_code"]= data["sky"].astype(float).where(data["sky"] != -9999, np.nan)

    # Precip: prefer 1-hr; handle trace(-1 => 0)
    p1 = data["p1"].astype(float)
    p6 = data["p6"].astype(float)
    p1 = p1.where(p1 != -9999, np.nan)
    p6 = p6.where(p6 != -9999, np.nan)
    p1 = p1.where(p1 != -1, 0.0) / 10.0
    p6 = p6.where(p6 != -1, 0.0) / 10.0

    # Hourly precip estimate: use p1 if present else p6/6
    data["prcp_mm"] = np.where(~np.isnan(p1), p1, np.where(~np.isnan(p6), p6/6.0, np.nan))

    return data[["dt","temp_c","dew_c","slp_hpa","wspd_ms","sky_code","prcp_mm"]]

def describe_week(row: pd.Series) -> str:
    t = row.get("temp_mean_c", np.nan)
    tmin = row.get("temp_p10_c", np.nan)
    tmax = row.get("temp_p90_c", np.nan)
    pr = row.get("prcp_week_mm", np.nan)
    prh = row.get("prcp_hours_pct", np.nan)
    wind = row.get("wind_mean_ms", np.nan)
    sky = row.get("sky_mode", np.nan)

    # simple buckets
    def temp_bucket(x):
        if np.isnan(x): return "variable temps"
        if x < 0: return "freezing"
        if x < 8: return "cold"
        if x < 15: return "cool"
        if x < 22: return "mild"
        if x < 28: return "warm"
        return "hot"

    def sky_bucket(code):
        if np.isnan(code): return "mixed skies"
        # 0-2 mostly clear, 3-5 partly, 6-8 mostly cloudy, 9+ obscured/other
        if code <= 2: return "mostly clear"
        if code <= 5: return "partly cloudy"
        if code <= 8: return "mostly cloudy"
        return "often obscured"

    def pr_bucket(pct):
        if np.isnan(pct): return "unknown rain risk"
        if pct < 5: return "very low rain risk"
        if pct < 15: return "low rain risk"
        if pct < 30: return "moderate rain risk"
        return "high rain risk"

    def wind_bucket(x):
        if np.isnan(x): return "variable winds"
        if x < 3: return "light winds"
        if x < 7: return "breezy"
        return "windy"

    return (
        f"{temp_bucket(t)} week (typical {tmin:.1f}–{tmax:.1f}°C), "
        f"{sky_bucket(sky)}, {pr_bucket(prh)} "
        f"(~{pr:.0f} mm/wk), {wind_bucket(wind)}."
    )

def pick_stations_for_query(
    query: str,
    stations: pd.DataFrame,
    inventory: pd.DataFrame,
    max_stations: int = 3,
    radius_km: float = 60.0
) -> Tuple[pd.DataFrame, str]:
    """
    Strategy:
    1) If query matches a country name in country-list.txt -> pick top stations in that country by record length.
    2) Else geocode query to lat/lon -> pick nearest stations within radius (fallback: nearest N).
    3) Also try direct name match for airports/stations.
    """
    q = query.strip().lower()
    countries = load_country_list()

    # Country match
    if q in countries:
        ctry = countries[q]
        cand = stations[stations["CTRY"].astype(str).str.upper() == ctry].copy()
        if cand.empty:
            return cand, f"Country match ({ctry}) but no stations found in isd-history."
        # Count available years per station from inventory
        inv = inventory.copy()
        inv["KEY"] = inv["USAF"].astype(str).str.zfill(6) + "-" + inv["WBAN"].astype(str).str.zfill(5)
        yrs = inv.groupby("KEY")["YEAR"].nunique().rename("n_years").reset_index()
        cand["KEY"] = cand["USAF"].astype(int).astype(str).str.zfill(6) + "-" + cand["WBAN"].astype(int).astype(str).str.zfill(5)
        cand = cand.merge(yrs, on="KEY", how="left").fillna({"n_years": 0})
        cand = cand.sort_values(["n_years"], ascending=False).head(max_stations)
        return cand, f"Country match: {query} (CTRY={ctry}); selected {len(cand)} stations with longest records."

    # Strong name match first (airport/station text)
    name_cand = stations[stations["STATION NAME"].astype(str).str.lower().str.contains(re.escape(q), na=False)].copy()
    if not name_cand.empty:
        # prefer the longest-record among name matches
        inv = inventory.copy()
        inv["KEY"] = inv["USAF"].astype(str).str.zfill(6) + "-" + inv["WBAN"].astype(str).str.zfill(5)
        yrs = inv.groupby("KEY")["YEAR"].nunique().rename("n_years").reset_index()
        name_cand["KEY"] = name_cand["USAF"].astype(int).astype(str).str.zfill(6) + "-" + name_cand["WBAN"].astype(int).astype(str).str.zfill(5)
        name_cand = name_cand.merge(yrs, on="KEY", how="left").fillna({"n_years": 0})
        name_cand = name_cand.sort_values(["n_years"], ascending=False).head(max_stations)
        return name_cand, f"Name match: selected {len(name_cand)} station(s) matching '{query}'."

    # Geocode then nearest stations
    geo = geocode_location(query)
    if not geo:
        return stations.head(0), "Could not geocode the location string."
    lat, lon, addr = geo
    cand = stations.copy()
    cand["dist_km"] = cand.apply(lambda r: haversine_km(lat, lon, r["LAT"], r["LON"]), axis=1)
    within = cand[cand["dist_km"] <= radius_km].sort_values("dist_km").head(max_stations)
    if within.empty:
        within = cand.sort_values("dist_km").head(max_stations)
        return within, f"Geocoded '{query}' to {addr}; no stations within {radius_km} km, using nearest {len(within)}."
    return within, f"Geocoded '{query}' to {addr}; using {len(within)} station(s) within {radius_km} km."

def available_years_for_station(inventory: pd.DataFrame, usaf: int, wban: int) -> List[int]:
    inv = inventory[(inventory["USAF"] == usaf) & (inventory["WBAN"] == wban)]
    years = sorted(inv["YEAR"].dropna().astype(int).unique().tolist())
    return years

def build_weekly_climatology(all_obs: pd.DataFrame) -> pd.DataFrame:
    """
    all_obs: combined hourly observations across stations/years.
    Returns weekly climatology (ISO week).
    """
    df = all_obs.copy()
    # ISO week
    iso = df["dt"].dt.isocalendar()
    df["iso_week"] = iso.week.astype(int)

    # hourly precip indicator
    df["prcp_pos"] = df["prcp_mm"].fillna(0) > 0.0

    # weekly aggregates for climatology (across all years)
    grp = df.groupby("iso_week")

    out = pd.DataFrame({
        "week": sorted(df["iso_week"].dropna().unique().tolist())
    }).set_index("week")

    out["temp_mean_c"] = grp["temp_c"].mean()
    out["temp_p10_c"] = grp["temp_c"].quantile(0.10)
    out["temp_p90_c"] = grp["temp_c"].quantile(0.90)
    out["dew_mean_c"]  = grp["dew_c"].mean()
    out["slp_mean_hpa"]= grp["slp_hpa"].mean()
    out["wind_mean_ms"]= grp["wspd_ms"].mean()
    out["wind_p90_ms"] = grp["wspd_ms"].quantile(0.90)

    # precip
    # Weekly total: sum of hourly estimates; this is an approximation (see note in UI)
    out["prcp_week_mm"] = grp["prcp_mm"].sum(min_count=1)
    out["prcp_hours_pct"] = grp["prcp_pos"].mean() * 100.0

    # cloudiness mode-ish
    def mode_series(s: pd.Series):
        s = s.dropna().astype(int)
        if s.empty: return np.nan
        return int(s.value_counts().idxmax())
    out["sky_mode"] = grp["sky_code"].apply(mode_series)

    out = out.reset_index()

    # Add “Expected weather” narrative
    out["expected_weather"] = out.apply(describe_week, axis=1)

    # Pretty columns
    out["temp_mean_c"] = out["temp_mean_c"].round(1)
    out["temp_p10_c"]  = out["temp_p10_c"].round(1)
    out["temp_p90_c"]  = out["temp_p90_c"].round(1)
    out["dew_mean_c"]  = out["dew_mean_c"].round(1)
    out["slp_mean_hpa"]= out["slp_mean_hpa"].round(1)
    out["wind_mean_ms"]= out["wind_mean_ms"].round(1)
    out["wind_p90_ms"] = out["wind_p90_ms"].round(1)
    out["prcp_week_mm"]= out["prcp_week_mm"].round(0)
    out["prcp_hours_pct"] = out["prcp_hours_pct"].round(1)

    # Ensure week 1..53 rows exist if missing (optional)
    # out = out.set_index("week").reindex(range(1, 54)).reset_index()

    return out.sort_values("week")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ISD Weekly Weather Patterns", layout="wide")

st.title("Weekly Weather Patterns (NOAA/NCEI ISD-Lite)")

with st.expander("How it works (important notes)", expanded=False):
    st.markdown(
        """
- Uses NOAA/NCEI **ISD-Lite** hourly station files (one file per station per year).  
- Station metadata: `isd-history.csv`; availability by year: `isd-inventory.csv`.  
- For a location string:
  - if it matches a country name (e.g., **Ireland**), it selects a few stations in that country with the **longest records**
  - otherwise it geocodes the text and selects the **nearest** stations
- Builds a **weekly climatology** (ISO week): aggregates all hours across all available years.
- **Precip caveat:** ISD-Lite provides 1-hour and 6-hour accumulations; the app uses 1-hour when present, else approximates using 6-hour totals / 6.
        """
    )

query = st.text_input("Location", value="Heathrow Airport")

colA, colB, colC = st.columns(3)
max_stations = colA.slider("Stations to blend", 1, 5, 3)
radius_km = colB.slider("Search radius (km) for geocoded locations", 10, 300, 60)
max_years = colC.slider("Max years per station to download (0 = all)", 0, 200, 0)

run = st.button("Generate weekly table")

if run:
    with st.spinner("Loading station lists..."):
        stations = load_station_history()
        inventory = load_inventory()

    chosen, rationale = pick_stations_for_query(
        query=query,
        stations=stations,
        inventory=inventory,
        max_stations=max_stations,
        radius_km=float(radius_km)
    )

    st.info(rationale)

    if chosen.empty:
        st.error("No stations selected. Try a different location string.")
        st.stop()

    # Show selected stations
    show_cols = ["USAF","WBAN","STATION NAME","CTRY","STATE","LAT","LON"]
    extra = [c for c in ["dist_km","n_years"] if c in chosen.columns]
    st.subheader("Selected stations")
    st.dataframe(chosen[show_cols + extra].reset_index(drop=True), use_container_width=True)

    all_obs = []
    with st.spinner("Downloading + parsing ISD-Lite files (cached locally)..."):
        prog = st.progress(0.0)
        tasks = []
        for _, r in chosen.iterrows():
            usaf = int(r["USAF"])
            wban = int(r["WBAN"])
            years = available_years_for_station(inventory, usaf, wban)
            if max_years and max_years > 0:
                # pick the most recent N years (fast) OR oldest? user asked “as much as possible”
                # We'll pick ALL unless limited; if limited, choose all spread out by taking recent.
                years = years[-max_years:]
            for y in years:
                tasks.append((usaf, wban, y))

        total = max(len(tasks), 1)
        done = 0

        for usaf, wban, y in tasks:
            path = download_isd_lite(usaf, wban, y)
            done += 1
            prog.progress(done / total)

            if path is None:
                continue
            try:
                df = parse_isd_lite_gz(path)
                all_obs.append(df)
            except Exception:
                # Skip corrupt/odd files
                continue

    if not all_obs:
        st.error("No ISD-Lite data could be loaded for the selected stations/years.")
        st.stop()

    obs = pd.concat(all_obs, ignore_index=True)
    # Drop rows with no key variables
    obs = obs.dropna(subset=["temp_c"], how="all")

    st.success(f"Loaded {len(obs):,} hourly observations.")

    weekly = build_weekly_climatology(obs)

    st.subheader("Weekly expected weather (climatology)")
    st.dataframe(
        weekly.rename(columns={
            "week": "ISO week",
            "temp_mean_c": "Temp mean (°C)",
            "temp_p10_c": "Temp p10 (°C)",
            "temp_p90_c": "Temp p90 (°C)",
            "dew_mean_c": "Dew mean (°C)",
            "slp_mean_hpa": "SLP mean (hPa)",
            "wind_mean_ms": "Wind mean (m/s)",
            "wind_p90_ms": "Wind p90 (m/s)",
            "prcp_week_mm": "Precip (mm/week)",
            "prcp_hours_pct": "Hours w/ precip (%)",
            "sky_mode": "Sky code (mode)",
            "expected_weather": "Expected weather",
        }),
        use_container_width=True,
        hide_index=True
    )

    # Downloads
    csv = weekly.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="weekly_weather_patterns.csv", mime="text/csv")
