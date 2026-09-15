# MnVision AI — Manganese Mining Intelligence Platform
### SIH26009 Prototype — Nagpur–Bhandara–Balaghat Manganese Belt

MnVision AI is a working prototype that pairs **two real, trained ML models**
with a **live FastAPI backend** and a **light, enterprise-grade web dashboard**
to support manganese exploration and production-risk decisions for a single,
deeply-modelled area: the **Nagpur–Bhandara–Balaghat manganese belt**
(Maharashtra / Madhya Pradesh, India) — the Sausar Group / Gondite Formation
belt that MOIL Ltd operates in.

```
Satellite/Geospatial Data → ML Model → Prospectivity Map
Production + Weather + Equipment Data → ML Model → Shortfall Risk → Explanation → Recommended Action → What-if
```

Everything shown in the frontend is computed by the trained models running in
the backend at request time — **there are no hardcoded or fake predictions.**

---

## 1. What's in this ZIP

```
mnvision/
├── ml/                          # Training pipeline
│   ├── MnVision_AI_Training.ipynb   ← Open this in Google Colab
│   ├── generate_data.py             ← Same logic as a standalone script
│   ├── train_prospectivity_model.py
│   ├── train_production_model.py
│   ├── data/                        ← Generated datasets (CSV/JSON)
│   └── models/                      ← Trained model artefacts (.joblib/.json)
│
├── backend/                     # FastAPI inference server
│   ├── app/
│   │   ├── main.py                  ← FastAPI app + routers
│   │   ├── inference.py             ← Loads models, runs REAL inference
│   │   ├── schemas.py
│   │   ├── models/                  ← Copy of trained artefacts (already included)
│   │   └── routers/                 ← prospectivity, production, alerts, report
│   └── requirements.txt
│
├── frontend/                    # Static HTML/CSS/JS dashboard
│   ├── index.html
│   ├── css/style.css                ← Light enterprise design system
│   ├── js/                          ← API calls, map, charts, what-if, alerts
│   └── vendor/                      ← Leaflet + Chart.js (bundled, no CDN needed)
│
└── README.md                    (this file)
```

## 2. Quick start (run the working prototype)

The trained models are **already included** under `backend/app/models/` — you
do not have to run the notebook first. Two terminals:

### Terminal 1 — Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000/docs` to see/try the live API (Swagger UI).

### Terminal 2 — Frontend
```bash
cd frontend
python3 -m http.server 8080
```
Open `http://localhost:8080` in your browser.

The dashboard talks to the backend at `http://localhost:8000` by default. If
you run the backend on a different host/port, set it before the page loads:
edit `frontend/js/config.js` (`window.MNVISION_API_BASE`), or add
```html
<script>window.MNVISION_API_BASE = "http://your-backend-host:8000";</script>
```
just above the other `<script>` tags in `index.html`.

## 3. Re-training the models (optional)

Everything needed to reproduce training from scratch is in `ml/`:

**Option A — Google Colab (recommended for the judges):**
Upload `ml/MnVision_AI_Training.ipynb` to Colab and run all cells top-to-bottom.
It regenerates the datasets, retrains both models, prints evaluation metrics
and plots, and produces downloadable `.joblib`/`.json` artefacts.

**Option B — Locally:**
```bash
cd ml
pip install -r ../backend/requirements.txt scikit-learn pandas numpy joblib matplotlib seaborn
python3 generate_data.py
python3 train_prospectivity_model.py
python3 train_production_model.py
# then copy ml/models/*.joblib and *.json into backend/app/models/
```

## 4. The two ML models

### 4.1 Manganese Prospectivity Model
- **Type:** RandomForestClassifier (probabilistic, `predict_proba`)
- **Labels:** presence/background — "prospective" cells are within 4 km of a
  real, documented manganese occurrence in the Sausar belt (Gumgaon, Kandri,
  Munsar, Beldongri, Chikla, Tirodi, Ukwa, Balaghat, Sitasaongi, Sitapatore,
  etc.); "background" cells are >8 km away. This is the standard
  presence/background approach used in real mineral-prospectivity mapping.
- **Features:** Fe-oxide band-ratio index, aeromagnetic anomaly, structural
  lineament density, NDVI, terrain slope, soil Mn/Fe geochemical anomaly,
  Gondite lithology proximity, distance to the Sausar belt structural axis.
- **Performance:** test ROC-AUC ≈ 0.998, 5-fold CV ROC-AUC ≈ 0.996 ± 0.004,
  Average Precision ≈ 0.93.
- **Output:** a 0–1 prospectivity score per grid cell, a Low/Medium/High zone,
  a model-confidence estimate (agreement across the forest's trees), and a
  transparent "why" breakdown (z-score of each feature vs the population,
  weighted by the model's global feature importance).

### 4.2 Production Shortfall Prediction Model
- **Type:** RandomForestRegressor (predicted tonnes) + RandomForestClassifier
  (shortfall probability)
- **Label:** an explicit, auditable business rule — `shortfall_event = 1` if
  actual production < 90% of the planned monthly target.
- **Features:** rainfall, temperature, soil moisture, equipment availability,
  downtime hours, labour availability, planned target, previous-month
  efficiency, calendar month, mine type.
- **Performance:** regression R² ≈ 0.94 (5-fold CV 0.955 ± 0.004, MAPE ≈ 10%);
  classifier ROC-AUC ≈ 0.89 (5-fold CV 0.94 ± 0.035).
- **Output:** predicted production, predicted shortfall (tonnes and %),
  shortfall probability, risk level, ranked risk drivers, and concrete
  recommended actions generated from which drivers are currently pushing
  risk up.

## 5. Data provenance — please read

This is a hackathon prototype built without direct access to authenticated
GIS/geological data portals or MOIL's internal systems. To keep the
**methodology real and defensible**:

- **Positive prospectivity labels** are anchored on the approximate
  coordinates of **real, named, publicly documented** MOIL/Sausar-belt
  manganese mining localities — not arbitrary points.
- **Geospatial feature values** (band ratios, magnetic anomaly, lithology
  proximity, lineaments, NDVI, slope, soil geochemistry) are **synthesised
  from geologically-informed functions of distance to these real occurrences
  and to the mapped Sausar-belt structural trend** — the standard approach
  when full raster stacks (Sentinel-2/ASTER, GSI Bhukosh, aeromagnetic grids,
  DEM) cannot be downloaded in a sandboxed environment. `ml/generate_data.py`
  and the notebook mark, feature by feature, exactly which real public
  dataset would replace each synthetic proxy.
- **Production/weather/equipment data** is representative data modelled on
  typical open-cast/underground manganese mining parameters and Central
  India's monsoon climatology — a clearly-labelled stand-in for real MOIL
  SCADA/ERP data. The schema is designed so real operational exports can be
  substituted directly, with no changes to feature engineering, training, or
  the backend/frontend.
- The shortfall label is an **explicit, auditable rule**, not an arbitrary tag.

**The prospectivity score is an exploration-prioritisation ranking derived
from surface geological/geospatial proxies. It does not directly detect
underground ore and must be confirmed through field validation (mapping,
sampling, drilling).**

## 6. API reference (backend)

| Endpoint | Description |
|---|---|
| `GET /api/prospectivity/summary` | Area info, zone counts, model metrics, feature importances |
| `GET /api/prospectivity/grid` | All scored grid cells (optionally bbox-filtered) |
| `GET /api/prospectivity/point?lat=&lon=` | Nearest-cell score, zone, confidence, explanation |
| `GET /api/prospectivity/occurrences` | Known documented manganese occurrences |
| `GET /api/production/mines` | List of mines in the study area |
| `GET /api/production/history?mine_id=` | Historical monthly production vs target |
| `GET /api/production/predict?mine_id=` | Live prediction: production, shortfall, risk, drivers, recommendations |
| `POST /api/production/whatif` | Re-run the model with overridden conditions |
| `GET /api/production/metrics` | Model validation metrics |
| `GET /api/alerts` | Alerts derived live from current model outputs |
| `GET /api/report` | Plain-text decision report assembled from live model outputs |

Full interactive docs: `http://localhost:8000/docs`

## 7. Design notes

The frontend follows a light, professional enterprise-analytics visual
language: white/off-white backgrounds, soft blue-grey panels, subtle borders,
dark navy/charcoal text, and meaningful colour coding (blue = information,
green = safe, amber = warning, red = critical). No dark mode, no neon, no
glow effects, no cyberpunk styling.

## 8. Known map-related notes

- The basemap uses OpenStreetMap's free standard tile server (no API key
  required, ever). If your network blocks external tile requests entirely,
  the map area gracefully falls back to a plain light-grey background after
  a few failed tile loads — the data layer (grid points, mine markers,
  click-to-select) keeps working either way.
- Grid points are small by design (to show thousands of cells clearly), but
  each has a generous invisible click-radius, so you don't need pixel-perfect
  accuracy — click anywhere near a point. For dense clusters (e.g. right
  around a documented mine), zoom in first for precise selection between
  adjacent cells.

## 9. Future enhancements (explicitly out of scope for this prototype)

- Reinforcement learning for adaptive production scheduling
- Digital twin simulation of mine operations
- Deep learning on true multispectral/hyperspectral raster stacks
- Direct ingestion of MOIL SCADA/ERP and GSI Bhukosh geo-databases once
  credentials/access are available (the pipeline is already shaped for this)
