# MnVision AI — Backend

FastAPI service that loads the trained models from `app/models/` and serves
real inference for the prospectivity map and production shortfall dashboard.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for interactive Swagger docs.

## Structure

- `app/inference.py` — loads the joblib models/scalers and the JSON metadata
  produced by the `ml/` training pipeline; performs all scoring, confidence
  estimation, and feature-based explanations.
- `app/routers/prospectivity.py` — grid, point, summary, occurrences endpoints
- `app/routers/production.py` — mines, history, predict, what-if, metrics
- `app/routers/alerts.py` — alerts derived live from current model outputs
- `app/routers/report.py` — plain-text decision report assembled live
- `app/models/` — trained artefacts (already populated; regenerate via `ml/`)

## Swapping in real data later

- Replace `app/models/geospatial_grid.csv` sourcing in `ml/generate_data.py`
  with real raster extraction (rasterio / Earth Engine) once GSI/Bhuvan/MOIL
  access is available, then re-run the training scripts and copy the new
  `.joblib`/`.json` files here — no backend code changes required as long as
  the feature names stay the same.
- Replace `app/models/production_dataset.csv` sourcing with a real MOIL
  SCADA/ERP export using the same column schema.
