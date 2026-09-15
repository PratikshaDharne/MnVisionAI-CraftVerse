"""
MnVision AI — Manganese Prospectivity Model Training
======================================================
Trains a probabilistic classifier that ranks locations in the
Nagpur–Bhandara–Balaghat manganese belt by prospectivity, using
geological/geophysical/remote-sensing-style features engineered
in generate_data.py.

Labels come from a presence/background scheme anchored on real, named,
documented manganese occurrences (see generate_data.py header) — NOT
arbitrary or fake labels.

Output artefacts (consumed directly by the FastAPI backend):
  models/prospectivity_model.joblib   - trained RandomForestClassifier
  models/prospectivity_scaler.joblib  - StandardScaler fit on training features
  models/prospectivity_meta.json      - feature list, metrics, feature stats
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix, brier_score_loss)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "geospatial_grid.csv")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "lithology_gondite_index",
    "ferrous_oxide_index",
    "magnetic_anomaly_norm",
    "structural_lineament_density",
    "ndvi",
    "slope_deg",
    "soil_mn_anomaly",
    "dist_to_belt_axis_km",
]

FEATURE_LABELS = {
    "lithology_gondite_index": "Gondite/Sausar lithology proximity",
    "ferrous_oxide_index": "Satellite Fe-oxide band-ratio index",
    "magnetic_anomaly_norm": "Aeromagnetic anomaly strength",
    "structural_lineament_density": "Structural lineament (fault/shear) density",
    "ndvi": "Vegetation index (NDVI)",
    "slope_deg": "Terrain slope",
    "soil_mn_anomaly": "Soil Mn/Fe geochemical anomaly",
    "dist_to_belt_axis_km": "Distance to Sausar belt structural axis (km)",
}


def main():
    df = pd.read_csv(DATA_PATH)
    labelled = df[df["prospectivity_label"] != -1].copy()
    print(f"Training rows: {len(labelled)}  "
          f"(positive={ (labelled.prospectivity_label==1).sum() }, "
          f"background={ (labelled.prospectivity_label==0).sum() })")

    X = labelled[FEATURES].values
    y = labelled["prospectivity_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ---- Evaluation ----
    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = clf.predict(X_test)

    auc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    brier = brier_score_loss(y_test, proba_test)
    report = classification_report(y_test, pred_test, output_dict=True)
    cm = confusion_matrix(y_test, pred_test).tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="roc_auc")

    print(f"Test ROC-AUC: {auc:.3f} | Avg Precision: {ap:.3f} | Brier: {brier:.3f}")
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(confusion_matrix(y_test, pred_test))

    # Refit on ALL labelled data for the deployed model (common practice once
    # cross-validated performance is confirmed acceptable)
    final_clf = RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    final_clf.fit(X_scaled, y)

    importances = dict(zip(FEATURES, final_clf.feature_importances_.tolist()))

    # feature population stats used later for per-point "why prospective" explanations
    feature_stats = {
        f: {"mean": float(labelled[f].mean()), "std": float(labelled[f].std() + 1e-9)}
        for f in FEATURES
    }

    joblib.dump(final_clf, os.path.join(MODEL_DIR, "prospectivity_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "prospectivity_scaler.joblib"))

    # Directional sign of each feature's relationship with the POSITIVE class,
    # from real correlation in the training data. Used at inference time so
    # local explanations correctly say "increases prospectivity" vs "decreases
    # prospectivity" regardless of whether the raw feature is positively or
    # negatively related to the target (e.g. distance-type features are
    # inversely related: smaller distance -> more prospective).
    direction_signs = {}
    for f in FEATURES:
        corr = np.corrcoef(labelled[f].values, labelled["prospectivity_label"].values)[0, 1]
        direction_signs[f] = 1.0 if (np.isnan(corr) or corr >= 0) else -1.0

    meta = {
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "feature_importances": importances,
        "feature_stats": feature_stats,
        "direction_signs": direction_signs,
        "metrics": {
            "test_roc_auc": auc,
            "test_average_precision": ap,
            "test_brier_score": brier,
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "confusion_matrix": cm,
            "classification_report": report,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "zone_thresholds": {"low_max": 0.35, "medium_max": 0.65},
        "model_type": "RandomForestClassifier",
        "trained_on": "generate_data.py::geospatial_grid.csv (presence/background labels, "
                      "positive_buffer=3km, background_buffer=9km around documented "
                      "Sausar-belt manganese occurrences)",
    }
    with open(os.path.join(MODEL_DIR, "prospectivity_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved: prospectivity_model.joblib, prospectivity_scaler.joblib, prospectivity_meta.json")
    print("Feature importances:", json.dumps(importances, indent=2))


if __name__ == "__main__":
    main()
