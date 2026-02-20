import os
import re
import math
import gzip
import io
import time
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="ISD Weekly Weather Patterns", layout="wide")


# -----------------------------
# NOAA / NCEI endpoints (ISD)
# -----------------------------
NCEI_NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/noaa"
ISD_HISTORY_URL = f"{NCEI_NOAA_BASE}/isd-history.csv"
ISD_INVENTORY_URL = f"{NCEI_NOAA_BASE}/isd-inventory.csv"
COUNTRY_LIST_URL = f"{NCEI_NOAA_BASE}/country-list.txt"
ISD_LITE_BASE = f"{NCEI_NOAA_BASE}/isd-lite"

# Streamlit Cloud writable cache dir
CACHE_DIR = os.path.join("/tmp", "cache_isd")
os.makedirs(CACHE_DIR, exist_ok=True)


# -----------------------------
# HTTP session (connection pooling)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "isd-weekly-climatology-streamlit/1.1"})
    return s


# -----------------------------
# Helpers
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def normalize_id(series: pd.Series, width: int) -> pd.Series:
    """
    Normalize USAF/WBAN style identifiers to zero-padded strings.
    Handles ints, floats-as-strings (e.g., "12345.0"), blanks.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": "", "None": ""})
    return s.str.zfill(width)


def station_id(usaf, wban) -> str:
    return f"{str(usaf).zfill(6)}-{str(wban).zfill(5)}"


def isd_lite_url(usaf, wban, year: int) -> str:
    return f"{ISD_LITE_BASE}/{year}/{station_id(usaf, wban)}-{year}.gz"


def local_isd_lite_path(usaf, wban, year: int) -> str:
    return os.path.join(CACHE_DIR, "isd-lite", str(year), f"{station_id(usaf, wban)}-{year}.gz")


# -----------------------------
# Load metadata
# -----------------------------
@st.cache_data(show_spinner=False)
def load_station_history() -> pd.DataFrame:
    """
    Loads isd-history.csv and normalizes station IDs.
    """
    sess = get_http_session()
    r = sess.get(ISD_HISTORY_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.dropna(subset=["LAT", "LON"])

    # Normalize IDs as strings
    df["USAF"] = normalize_id(df["USAF"], 6)
    df["WBAN"] = normalize_id(df["WBAN"], 5)
    df["KEY"] = df["USAF"] + "-" + df["WBAN"]
    return df


@st.cache_data(show_spinner=False)
def load_inventory() -> pd.DataFrame:
    """
    Loads isd-inventory.csv and normalizes station IDs and year.
    """
    sess = get_http_session()
    r = sess.get(ISD_INVENTORY_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [c.strip().upper() for c in df.columns]

    df["USAF"] = normalize_id(df["USAF"], 6)
    df["WBAN"] = normalize_id(df["WBAN"], 5)
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["KEY"] = df["USAF"] + "-" + df["WBAN"]
    return df


@st.cache_data(show_spinner=False)
def load_country_list() -> Dict[str, str]:
    """
    Parses country-list.txt to mapping lower(name)->code.
    """
    sess = get_http_session()
    r = sess.get(COUNTRY_LIST_URL, timeout=60)
    r.raise_for_status()
    text = r.text.replace("\n", " ")
    tokens = text.split()

    out = {}
    i = 0
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


# -----------------------------
# Geocoding
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_geocoder() -> Nominatim:
    return Nominatim(user_agent="isd_weekly_climatology_app_streamlit_cloud")


def geocode_location(q: str) -> Optional[Tuple[float, float, str]]:
    """
    Robust geocoding with retries (Nominatim can time out / rate-limit).
    """
    geocoder = get_geocoder()
    for attempt in range(3):
        try:
            loc = geocoder.geocode(q, exactly_one=True, addressdetails=False, timeout=20)
            if not loc:
                return None
            return float(loc.latitude), float(loc.longitude), str(loc.address)
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(1.5 * (attempt + 1))
    return None


# -----------------------------
# Station selection + inventory
# -----------------------------
def available_years_for_station(inventory: pd.DataFrame, usaf: str, wban: str) -> List[int]:
    key = station_id(usaf, wban)
    inv = inventory[inventory["KEY"] == key]
    years = sorted(inv["YEAR"].dropna().astype(int).unique().tolist())
    return years


def pick_stations_for_query(
    query: str,
    stations: pd.DataFrame,
    inventory: pd.DataFrame,
    max_stations: int = 3,
    radius_km: float = 60.0,
) -> Tuple[pd.DataFrame, str]:
    """
    Selection logic:
    - Country name -> choose longest-record stations in that country
    - Else station name match
    - Else geocode -> nearest stations
    """
    q = query.strip().lower()
    countries = load_country_list()

    # Country match
    if q in countries:
        ctry = countries[q]
        cand = stations[stations["CTRY"].astype(str).str.upper() == ctry].copy()
        if cand.empty:
            return cand, f"Country match ({ctry}) but no stations found in isd-history."

        yrs = inventory.groupby("KEY")["YEAR"].nunique().rename("n_years").reset_index()
        cand = cand.merge(yrs, on="KEY", how="left").fillna({"n_years": 0})
        cand = cand.sort_values(["n_years"], ascending=False).head(max_stations)
        return cand, f"Country match: {query} (CTRY={ctry}); selected {len(cand)} station(s) with longest records."

    # Station name match
    name_cand = stations[stations["STATION NAME"].astype(str).str.lower().str.contains(re.escape(q), na=False)].copy()
    if not name_cand.empty:
        yrs = inventory.groupby("KEY")["YEAR"].nunique().rename("n_years").reset_index()
        name_cand = name_cand.merge(yrs, on="KEY", how="left").fillna({"n_years": 0})
        name_cand = name_cand.sort_values(["n_years"], ascending=False).head(max_stations)
        return name_cand, f"Name match: selected {len(name_cand)} station(s) matching '{query}'."

    # Geocode + nearest
    geo = geocode_location(query)
    if not geo:
        return stations.head(0), "Could not geocode the location string. Try 'Heathrow Airport, London' or 'Dublin, Ireland'."

    lat, lon, addr = geo
    cand = stations.copy()
    cand["dist_km"] = cand.apply(lambda r: haversine_km(lat, lon, r["LAT"], r["LON"]), axis=1)

    within = cand[cand["dist_km"] <= radius_km].sort_values("dist_km").head(max_stations)
    if within.empty:
        within = cand.sort_values("dist_km").head(max_stations)
        return within, f"Geocoded '{query}' to {addr}; no stations within {radius_km} km, using nearest {len(within)}."

    return within, f"Geocoded '{query}' to {addr}; using {len(within)} station(s) within {radius_km} km."


# -----------------------------
# Download + parse ISD-Lite
# -----------------------------
def download_isd_lite(usaf: str, wban: str, year: int) -> Optional[str]:
    """
    Downloads the .gz if missing. Returns local path, or None if 404.
    """
    path = local_isd_lite_path(usaf, wban, year)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    url = isd_lite_url(usaf, wban, year)
    sess = get_http_session()
    r = sess.get(url, stream=True, timeout=90)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    return path


def parse_isd_lite_gz(path: str) -> pd.DataFrame:
    """
    Robust ISD-Lite parser using whitespace-separated columns.
    Columns:
    year month day hour temp dew slp wdir wspd sky p1 p6
    Values scaled by 10; missing = -9999; trace precip = -1
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        data = pd.read_csv(
            f,
            sep=r"\s+",
            header=None,
            names=["year", "month", "day", "hour", "temp", "dew", "slp", "wdir", "wspd", "sky", "p1", "p6"],
            engine="python",
        )

    # datetime (UTC)
    data["dt"] = pd.to_datetime(
        data[["year", "month", "day", "hour"]].astype("int64"),
        errors="coerce",
        utc=True,
    )
    data = data.dropna(subset=["dt"])

    def scale(series: pd.Series, factor: float) -> pd.Series:
        x = pd.to_numeric(series, errors="coerce").astype(float)
        x = x.where(x != -9999, np.nan)
        return x / factor

    data["temp_c"] = scale(data["temp"], 10.0)
    data["dew_c"] = scale(data["dew"], 10.0)
    data["slp_hpa"] = scale(data["slp"], 10.0)
    data["wspd_ms"] = scale(data["wspd"], 10.0)
    data["sky_code"] = pd.to_numeric(data["sky"], errors="coerce").astype(float).where(data["sky"] != -9999, np.nan)

    # precip
    p1 = pd.to_numeric(data["p1"], errors="coerce").astype(float).where(data["p1"] != -9999, np.nan)
    p6 = pd.to_numeric(data["p6"], errors="coerce").astype(float).where(data["p6"] != -9999, np.nan)
    p1 = p1.where(p1 != -1, 0.0) / 10.0
    p6 = p6.where(p6 != -1, 0.0) / 10.0

    data["prcp_mm"] = np.where(~np.isnan(p1), p1, np.where(~np.isnan(p6), p6 / 6.0, np.nan))

    return data[["dt", "temp_c", "dew_c", "slp_hpa", "wspd_ms", "sky_code", "prcp_mm"]]


# -----------------------------
# Climatology calculations
# -----------------------------
def describe_week(row: pd.Series) -> str:
    t = row.get("temp_mean_c", np.nan)
    tmin = row.get("temp_p10_c", np.nan)
    tmax = row.get("temp_p90_c", np.nan)
    pr = row.get("prcp_week_mm", np.nan)
    prh = row.get("prcp_hours_pct", np.nan)
    wind = row.get("wind_mean_ms", np.nan)
    sky = row.get("sky_mode", np.nan)

    def temp_bucket(x):
        if np.isnan(x):
            return "variable temps"
        if x < 0:
            return "freezing"
        if x < 8:
            return "cold"
        if x < 15:
            return "cool"
        if x < 22:
            return "mild"
        if x < 28:
            return "warm"
        return "hot"

    def sky_bucket(code):
        if np.isnan(code):
            return "mixed skies"
        if code <= 2:
            return "mostly clear"
        if code <= 5:
            return "partly cloudy"
        if code <= 8:
            return "mostly cloudy"
        return "often obscured"

    def pr_bucket(pct):
        if np.isnan(pct):
            return "unknown rain risk"
        if pct < 5:
            return "very low rain risk"
        if pct < 15:
            return "low rain risk"
        if pct < 30:
            return "moderate rain risk"
        return "high rain risk"

    def wind_bucket(x):
        if np.isnan(x):
            return "variable winds"
        if x < 3:
            return "light winds"
        if x < 7:
            return "breezy"
        return "windy"

    tmin_s = f"{tmin:.1f}" if not np.isnan(tmin) else "?"
    tmax_s = f"{tmax:.1f}" if not np.isnan(tmax) else "?"
    pr_s = f"{pr:.0f}" if not np.isnan(pr) else "?"

    return (
        f"{temp_bucket(t)} week (typical {tmin_s}–{tmax_s}°C), "
        f"{sky_bucket(sky)}, {pr_bucket(prh)} (~{pr_s} mm/wk), {wind_bucket(wind)}."
    )


def build_weekly_climatology(all_obs: pd.DataFrame) -> pd.DataFrame:
    df = all_obs.copy()
    iso = df["dt"].dt.isocalendar()
    df["iso_week"] = iso.week.astype(int)

    df["prcp_pos"] = df["prcp_mm"].fillna(0) > 0.0
    grp = df.groupby("iso_week")

    out = pd.DataFrame({"week": sorted(df["iso_week"].dropna().unique().tolist())}).set_index("week")

    out["temp_mean_c"] = grp["temp_c"].mean()
    out["temp_p10_c"] = grp["temp_c"].quantile(0.10)
    out["temp_p90_c"] = grp["temp_c"].quantile(0.90)
    out["dew_mean_c"] = grp["dew_c"].mean()
    out["slp_mean_hpa"] = grp["slp_hpa"].mean()
    out["wind_mean_ms"] = grp["wspd_ms"].mean()
    out["wind_p90_ms"] = grp["wspd_ms"].quantile(0.90)

    out["prcp_week_mm"] = grp["prcp_mm"].sum(min_count=1)
    out["prcp_hours_pct"] = grp["prcp_pos"].mean() * 100.0

    def mode_series(s: pd.Series):
        s = s.dropna().astype(int)
        if s.empty:
            return np.nan
        return int(s.value_counts().idxmax())

    out["sky_mode"] = grp["sky_code"].apply(mode_series)

    out = out.reset_index()
    out["expected_weather"] = out.apply(describe_week, axis=1)

    # round for display
    for c in ["temp_mean_c", "temp_p10_c", "temp_p90_c", "dew_mean_c", "slp_mean_hpa", "wind_mean_ms", "wind_p90_ms"]:
        out[c] = out[c].round(1)
    out["prcp_week_mm"] = out["prcp_week_mm"].round(0)
    out["prcp_hours_pct"] = out["prcp_hours_pct"].round(1)

    return out.sort_values("week")


# -----------------------------
# UI
# -----------------------------
st.title("Weekly Weather Patterns (NOAA/NCEI ISD-Lite)")

with st.expander("How it works (important notes)", expanded=False):
    st.markdown(
        """
- Uses NOAA/NCEI **ISD-Lite** hourly station files (one file per station per year).
- Station metadata from `isd-history.csv`; year availability from `isd-inventory.csv`.
- Location string handling:
  - If it matches a country name (e.g., **Ireland**), it selects stations in that country with the **longest records**.
  - Otherwise it attempts a station-name match, then falls back to geocoding + nearest stations.
- Weekly climatology uses ISO week numbers and aggregates all hours across all available years.
- **Precip caveat:** ISD-Lite provides 1-hour and 6-hour accumulations; this app uses 1-hour when present, else approximates via 6-hour totals / 6.
        """
    )

query = st.text_input("Location", value="Heathrow Airport")

colA, colB, colC = st.columns(3)
max_stations = colA.slider("Stations to blend", 1, 5, 3)
radius_km = colB.slider("Search radius (km) for geocoded locations", 10, 300, 60)

# Cloud-friendly default to avoid huge downloads; user can set 0 for all years
max_years = colC.slider("Max years per station (0 = all)", 0, 120, 30)

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
        radius_km=float(radius_km),
    )

    st.info(rationale)

    if chosen.empty:
        st.error("No stations selected. Try a different location string.")
        st.stop()

    show_cols = ["USAF", "WBAN", "STATION NAME", "CTRY", "STATE", "LAT", "LON"]
    extra = [c for c in ["dist_km", "n_years"] if c in chosen.columns]
    st.subheader("Selected stations")
    st.dataframe(chosen[show_cols + extra].reset_index(drop=True), use_container_width=True)

    all_obs = []
    with st.spinner("Downloading + parsing ISD-Lite files (cached while the server runs)..."):
        prog = st.progress(0.0)

        tasks = []
        for _, r in chosen.iterrows():
            usaf = str(r["USAF"]).zfill(6)
            wban = str(r["WBAN"]).zfill(5)
            years = available_years_for_station(inventory, usaf, wban)

            if max_years and max_years > 0:
                years = years[-max_years:]  # newest N years

            for y in years:
                tasks.append((usaf, wban, int(y)))

        # Debug: show how many station-years will be fetched
        st.write(f"Station-years to fetch: {len(tasks)}")
        if len(tasks) > 0:
            st.write("Example task:", tasks[0])

        total = max(len(tasks), 1)
        done = 0

        for usaf, wban, y in tasks:
            done += 1
            prog.progress(done / total)
            try:
                path = download_isd_lite(usaf, wban, y)
                if path is None:
                    continue
                df = parse_isd_lite_gz(path)
                all_obs.append(df)
            except Exception:
                continue

    if not all_obs:
        st.error("No ISD-Lite data could be loaded for the selected stations/years.")
        st.stop()

    obs = pd.concat(all_obs, ignore_index=True)

    # If temp is all missing, you'll see 0 after this
    obs = obs.dropna(subset=["temp_c"], how="all")

    st.success(f"Loaded {len(obs):,} hourly observations.")

    if len(obs) == 0:
        st.error(
            "Data loaded but contained no valid temperature observations after cleaning. "
            "Try a different station or reduce the filtering."
        )
        st.stop()

    weekly = build_weekly_climatology(obs)

    st.subheader("Weekly expected weather (climatology)")
    st.dataframe(
        weekly.rename(
            columns={
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
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv = weekly.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="weekly_weather_patterns.csv", mime="text/csv")
