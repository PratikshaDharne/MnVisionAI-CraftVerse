"""
MnVision AI — Data Generation Module
=====================================
Study Area: Nagpur–Bhandara–Balaghat Manganese Belt (Maharashtra / Madhya Pradesh, India)

This belt hosts the manganese ore deposits of the Sausar Group (Gondite Formation),
a Precambrian metasedimentary sequence which is the principal manganese-ore-bearing
horizon of central India and the operating ground of MOIL Ltd (Manganese Ore India
Limited) — the country's leading manganese producer.

IMPORTANT — DATA PROVENANCE NOTE (read this before using the numbers below):
-----------------------------------------------------------------------------
1. KNOWN_OCCURRENCES below are APPROXIMATE coordinates of publicly documented
   MOIL / Sausar-belt manganese mining centres (Nagpur, Bhandara, Balaghat
   districts). These anchor the "positive" class for prospectivity modelling.
   They are deliberately coarse (±a few km) because precise deposit polygons are
   proprietary GSI/MOIL survey data. When real GSI/MOIL geo-database access is
   available, replace `KNOWN_OCCURRENCES` with the authoritative point/polygon
   layer — the rest of the pipeline (feature engineering, training, inference)
   is unaffected.
2. The geospatial FEATURE VALUES (Fe-oxide band-ratio index, magnetic anomaly,
   lineament density, NDVI, slope, gondite lithology proximity) are SYNTHESISED
   using geologically-informed functions of distance-to-known-occurrence plus
   realistic noise. This mirrors the standard "presence/background" approach
   used in real mineral-prospectivity mapping (e.g. MaxEnt / weights-of-evidence
   studies) when full raster stacks (ASTER/Sentinel band ratios, aeromagnetic
   grids, geological maps) are not directly downloadable in this environment.
   The Colab notebook contains clearly marked cells showing exactly where to
   swap each synthetic feature for a real raster extraction
   (rasterio.sample / Google Earth Engine reduceRegion) once the following
   public sources are connected:
     - USGS/ASTER or Sentinel-2 band ratios (iron-oxide, ferrous-mineral index)
     - Bhukosh (GSI) lithology & structural lineament layers
     - National Geophysical Mapping (aeromagnetic) grids
     - Bhoonidhi / Bhuvan DEM for slope & terrain ruggedness
3. Production/weather/equipment data is REPRESENTATIVE OPERATIONAL DATA modelled
   on typical open-cast/underground manganese mining parameters (rainfall
   seasonality of central India, standard equipment-availability ranges,
   monsoon soil-moisture cycles). It is clearly a stand-in for real MOIL
   SCADA/ERP feeds. The schema (columns, units, ranges) is designed so that
   real MOIL operational exports can be dropped in with no pipeline changes.

Nothing here uses arbitrary/fake LABELS: prospectivity labels come from
distance to real, named, documented ore-occurrence localities; shortfall
labels come from an explicit, auditable business rule (actual < 90% of
planned target).
"""

import numpy as np
import pandas as pd
import json
import os

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 1. STUDY AREA & KNOWN MANGANESE OCCURRENCES (Sausar Belt, MOIL territory)
# -----------------------------------------------------------------------
AREA_NAME = "Nagpur–Bhandara–Balaghat Manganese Belt"
LAT_MIN, LAT_MAX = 20.75, 22.05
LON_MIN, LON_MAX = 78.75, 80.45

# Approximate, publicly-known MOIL / Sausar-belt manganese mining localities.
# Source: publicly available descriptions of MOIL's operating mines in the
# Nagpur (Maharashtra), Bhandara (Maharashtra) and Balaghat (MP) districts.
KNOWN_OCCURRENCES = [
    {"name": "Gumgaon Mine",        "lat": 21.020, "lon": 79.050},
    {"name": "Kandri Mine",         "lat": 21.055, "lon": 79.145},
    {"name": "Munsar Mine",         "lat": 21.145, "lon": 79.205},
    {"name": "Beldongri Mine",      "lat": 21.095, "lon": 79.780},
    {"name": "Chikla Mine",         "lat": 21.300, "lon": 79.900},
    {"name": "Dongri Buzurg Mine",  "lat": 21.250, "lon": 79.955},
    {"name": "Tirodi Mine",         "lat": 21.680, "lon": 80.100},
    {"name": "Ukwa Mine",           "lat": 21.755, "lon": 80.150},
    {"name": "Sitasaongi Mine",     "lat": 21.795, "lon": 80.195},
    {"name": "Balaghat Mine",       "lat": 21.810, "lon": 80.185},
    {"name": "Sitapatore Mine",     "lat": 21.770, "lon": 80.230},
    {"name": "Parseoni Occurrence", "lat": 21.310, "lon": 79.230},
]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def dist_to_nearest_occurrence(lat, lon):
    dists = [haversine_km(lat, lon, o["lat"], o["lon"]) for o in KNOWN_OCCURRENCES]
    return float(np.min(dists))


# The Sausar belt trends broadly NE–SW. We encode a "belt axis" so that
# lithology/structural features are stronger *along* the trend, not just
# near individual point occurrences (geologically realistic: gondite bands
# are elongated, not circular).
BELT_AXIS_START = np.array([20.95, 78.95])
BELT_AXIS_END = np.array([21.85, 80.30])


def dist_to_belt_axis_km(lat, lon):
    p = np.array([lat, lon])
    a, b = BELT_AXIS_START, BELT_AXIS_END
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    proj = a + t * ab
    return haversine_km(lat, lon, proj[0], proj[1])


# -----------------------------------------------------------------------
# 2. GEOSPATIAL GRID GENERATION (prospectivity modelling dataset)
# -----------------------------------------------------------------------
def build_geospatial_grid(n_lat=70, n_lon=78):
    lats = np.linspace(LAT_MIN, LAT_MAX, n_lat)
    lons = np.linspace(LON_MIN, LON_MAX, n_lon)
    rows = []
    cell_id = 0
    for la in lats:
        for lo in lons:
            d_occ = dist_to_nearest_occurrence(la, lo)
            d_belt = dist_to_belt_axis_km(la, lo)
            rows.append({"cell_id": cell_id, "lat": round(la, 5), "lon": round(lo, 5),
                         "dist_to_known_occurrence_km": d_occ,
                         "dist_to_belt_axis_km": d_belt})
            cell_id += 1
    df = pd.DataFrame(rows)
    return df


def engineer_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    d_occ = df["dist_to_known_occurrence_km"].values
    d_belt = df["dist_to_belt_axis_km"].values

    # Gondite/Sausar lithology proximity index (0-1): decays with distance to
    # belt axis, i.e. probability the pixel sits within mapped gondite/schist
    # lithology. Real source: GSI Bhukosh lithology polygons.
    lithology_gondite_index = np.clip(np.exp(-d_belt / 6.0) + rng.normal(0, 0.12, n), 0, 1)

    # Fe/Mn-oxide band-ratio index (satellite-derived spectral proxy).
    # Real source: Sentinel-2 / ASTER band ratio (e.g., (B4/B3) iron-oxide index).
    ferrous_oxide_index = np.clip(
        0.75 * np.exp(-d_occ / 4.0) + 0.25 * np.exp(-d_belt / 8.0)
        + rng.normal(0, 0.16, n), 0, 1.3)

    # Aeromagnetic anomaly strength (nT, normalised 0-1 for modelling).
    # Manganese-iron mineralisation in gondites often produces a measurable
    # magnetic response relative to background metasediments.
    magnetic_anomaly = np.clip(
        0.8 * np.exp(-d_occ / 5.5) + rng.normal(0, 0.20, n), 0, 1.2)

    # Structural lineament density (faults/shear-zones per unit area) —
    # manganese ore-bodies in the Sausar Group are structurally controlled.
    lineament_density = np.clip(
        0.6 * np.exp(-d_belt / 7.0) + 0.3 * rng.random(n) + rng.normal(0, 0.12, n), 0, 1)

    # NDVI — mineralised/rocky exposed terrain tends to have sparser vegetation
    # than surrounding agricultural land; noisy but weakly informative.
    ndvi = np.clip(0.55 - 0.20 * np.exp(-d_occ / 6.0) + rng.normal(0, 0.14, n), -0.1, 0.9)

    # Terrain ruggedness / slope (degrees) — ore-bearing ridges are often
    # topographically expressed as low hill ranges.
    slope_deg = np.clip(3 + 9 * np.exp(-d_belt / 9.0) + rng.normal(0, 2.6, n), 0, 30)

    # Soil geochemistry proxy — Mn/Fe soil anomaly (ppm-normalised 0-1)
    soil_mn_anomaly = np.clip(
        0.7 * np.exp(-d_occ / 3.5) + rng.normal(0, 0.16, n), 0, 1.2)

    df["lithology_gondite_index"] = lithology_gondite_index
    df["ferrous_oxide_index"] = ferrous_oxide_index
    df["magnetic_anomaly_norm"] = magnetic_anomaly
    df["structural_lineament_density"] = lineament_density
    df["ndvi"] = ndvi
    df["slope_deg"] = slope_deg
    df["soil_mn_anomaly"] = soil_mn_anomaly
    return df


def label_prospectivity(df: pd.DataFrame, positive_buffer_km=4.0, background_buffer_km=8.0):
    """
    Presence / background labelling (standard mineral-prospectivity approach):
      label = 1  -> within `positive_buffer_km` of a documented occurrence
      label = 0  -> further than `background_buffer_km` (true background)
      (points strictly between the two buffers are dropped from TRAINING to
       avoid ambiguous boundary labels — they still appear in the inference
       grid, just not in the labelled training set.)
    """
    df = df.copy()
    d = df["dist_to_known_occurrence_km"].values
    label = np.full(len(df), -1, dtype=int)  # -1 = ambiguous / unlabeled
    label[d <= positive_buffer_km] = 1
    label[d > background_buffer_km] = 0
    df["prospectivity_label"] = label
    return df


# -----------------------------------------------------------------------
# 3. PRODUCTION / OPERATIONAL DATASET (shortfall prediction)
# -----------------------------------------------------------------------
MINES = [
    {"mine_id": "MN-BAL-01", "name": "Balaghat Mine",  "monthly_target_t": 9200, "type": "underground"},
    {"mine_id": "MN-UKW-02", "name": "Ukwa Mine",       "monthly_target_t": 6400, "type": "underground"},
    {"mine_id": "MN-GUM-03", "name": "Gumgaon Mine",    "monthly_target_t": 5100, "type": "opencast"},
    {"mine_id": "MN-CHK-04", "name": "Chikla Mine",     "monthly_target_t": 4300, "type": "opencast"},
]

MONTH_RAIN_MEAN = {  # central-India monsoon-dominated rainfall climatology (mm/month)
    1: 12, 2: 15, 3: 18, 4: 22, 5: 30, 6: 165,
    7: 310, 8: 280, 9: 190, 10: 65, 11: 18, 12: 8,
}
MONTH_TEMP_MEAN = {  # deg C
    1: 21, 2: 25, 3: 31, 4: 37, 5: 40, 6: 35,
    7: 29, 8: 28, 9: 29, 10: 29, 11: 24, 12: 20,
}


def build_production_dataset(n_years=4, start_year=2021):
    rows = []
    for mine in MINES:
        base = mine["monthly_target_t"]
        # mine-specific equipment reliability baseline
        eq_baseline = rng.uniform(0.80, 0.93)
        for y in range(n_years):
            year = start_year + y
            for m in range(1, 13):
                rain = max(0, rng.normal(MONTH_RAIN_MEAN[m], MONTH_RAIN_MEAN[m] * 0.25 + 3))
                temp = rng.normal(MONTH_TEMP_MEAN[m], 2.0)
                # soil moisture responds to rainfall with 1-month memory-ish smoothing
                soil_moisture = np.clip(10 + 0.18 * rain + rng.normal(0, 4), 5, 95)
                equipment_availability = np.clip(
                    eq_baseline * 100 - 0.04 * rain + rng.normal(0, 4), 45, 99)
                downtime_hours = np.clip(
                    (100 - equipment_availability) * 1.1 + rng.normal(0, 6), 0, 220)
                labor_availability = np.clip(rng.normal(92, 5), 60, 100)
                planned_target = base * rng.uniform(0.97, 1.03)

                # ---- ground-truth production generating process ----
                # Calibrated so that a well-run month lands close to plan (efficiency ~1.0)
                # and shortfalls emerge from compounding adverse conditions (heavy monsoon
                # haulage disruption, equipment downtime, labour shortage, heat stress) —
                # not from an always-on penalty. This keeps the shortfall_event base rate
                # realistic (roughly a quarter to a third of months) rather than degenerate.
                equip_penalty = 0.55 * (1 - equipment_availability / 100)
                rain_penalty = 0.45 * max(0, rain - 150) / 150        # only heavy monsoon hurts haulage
                downtime_penalty = 0.28 * downtime_hours / 220
                labor_penalty = 0.22 * max(0, 88 - labor_availability) / 88
                temp_penalty = 0.12 * max(0, temp - 38) / 10

                efficiency = np.clip(
                    1.03 - equip_penalty - rain_penalty - downtime_penalty - labor_penalty
                    - temp_penalty + rng.normal(0, 0.04), 0.15, 1.08)

                actual_production = planned_target * efficiency
                rows.append({
                    "mine_id": mine["mine_id"], "mine_name": mine["name"],
                    "mine_type": mine["type"],
                    "year": year, "month": m,
                    "rainfall_mm": round(rain, 1),
                    "avg_temperature_c": round(temp, 1),
                    "soil_moisture_pct": round(soil_moisture, 1),
                    "equipment_availability_pct": round(equipment_availability, 1),
                    "downtime_hours": round(downtime_hours, 1),
                    "labor_availability_pct": round(labor_availability, 1),
                    "planned_target_tonnes": round(planned_target, 1),
                    "actual_production_tonnes": round(actual_production, 1),
                })
    df = pd.DataFrame(rows)
    df["shortfall_tonnes"] = df["planned_target_tonnes"] - df["actual_production_tonnes"]
    df["shortfall_pct"] = 100 * df["shortfall_tonnes"] / df["planned_target_tonnes"]
    # Explicit, auditable business rule for the "at-risk" binary label:
    # a month counts as a SHORTFALL EVENT if actual output trails plan by >10%.
    df["shortfall_event"] = (df["shortfall_pct"] > 10).astype(int)
    # lag feature: previous month production ratio (captures momentum)
    df = df.sort_values(["mine_id", "year", "month"]).reset_index(drop=True)
    df["prev_month_efficiency"] = df.groupby("mine_id")["actual_production_tonnes"].shift(1) / \
                                    df.groupby("mine_id")["planned_target_tonnes"].shift(1)
    df["prev_month_efficiency"] = df["prev_month_efficiency"].fillna(df["prev_month_efficiency"].mean())
    return df


def main():
    print(f"Building geospatial grid for {AREA_NAME} ...")
    grid = build_geospatial_grid()
    grid = engineer_geospatial_features(grid)
    grid = label_prospectivity(grid)
    grid_path = os.path.join(OUT_DIR, "geospatial_grid.csv")
    grid.to_csv(grid_path, index=False)
    print(f"  -> {grid_path}  ({len(grid)} cells, "
          f"{(grid.prospectivity_label==1).sum()} positive / "
          f"{(grid.prospectivity_label==0).sum()} background / "
          f"{(grid.prospectivity_label==-1).sum()} unlabeled-ambiguous)")

    with open(os.path.join(OUT_DIR, "known_occurrences.json"), "w") as f:
        json.dump({"area_name": AREA_NAME, "bbox": [LAT_MIN, LON_MIN, LAT_MAX, LON_MAX],
                   "occurrences": KNOWN_OCCURRENCES}, f, indent=2)

    print("Building production/operational dataset ...")
    prod = build_production_dataset()
    prod_path = os.path.join(OUT_DIR, "production_dataset.csv")
    prod.to_csv(prod_path, index=False)
    print(f"  -> {prod_path}  ({len(prod)} monthly records, "
          f"{(prod.shortfall_event==1).sum()} shortfall-events)")

    with open(os.path.join(OUT_DIR, "mines.json"), "w") as f:
        json.dump(MINES, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
