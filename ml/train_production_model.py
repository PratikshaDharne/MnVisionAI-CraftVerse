"""
MnVision AI — Production Shortfall Prediction Model Training
================================================================
Trains two complementary models on the operational dataset
(production, rainfall, soil moisture, temperature, equipment
availability/downtime, labour availability):

  1. A RandomForestRegressor predicting ACTUAL monthly production (tonnes)
     given planned target + operational/environmental conditions.
  2. A RandomForestClassifier predicting the PROBABILITY of a shortfall
     event (actual production < 90% of planned target for the month) —
     an explicit, auditable business rule, not an arbitrary label.

Both models share the same feature set so the backend can serve a single
coherent "predicted production + shortfall risk" response per request.

Output artefacts:
  models/production_regressor.joblib
  models/shortfall_classifier.joblib
  models/production_scaler.joblib
  models/production_meta.json
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                              r2_score, roc_auc_score, classification_report,
                              confusion_matrix)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "production_dataset.csv")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temperature_c",
    "soil_moisture_pct",
    "equipment_availability_pct",
    "downtime_hours",
    "labor_availability_pct",
    "planned_target_tonnes",
    "prev_month_efficiency",
    "month",
]
CATEGORICAL_FEATURES = ["mine_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "rainfall_mm": "Monthly rainfall (mm)",
    "avg_temperature_c": "Average temperature (°C)",
    "soil_moisture_pct": "Soil moisture (%)",
    "equipment_availability_pct": "Equipment availability (%)",
    "downtime_hours": "Equipment downtime (hours)",
    "labor_availability_pct": "Labour availability (%)",
    "planned_target_tonnes": "Planned production target (t)",
    "prev_month_efficiency": "Previous month efficiency ratio",
    "month": "Calendar month",
    "mine_type_encoded": "Mine type (opencast / underground)",
}


def main():
    df = pd.read_csv(DATA_PATH)

    le = LabelEncoder()
    df["mine_type_encoded"] = le.fit_transform(df["mine_type"])
    model_features = NUMERIC_FEATURES + ["mine_type_encoded"]

    X = df[model_features].values
    y_reg = df["actual_production_tonnes"].values
    y_clf = df["shortfall_event"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- Regression: predicted production ----------------
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_reg, test_size=0.25, random_state=42)
    reg = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=2,
                                 random_state=42, n_jobs=-1)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    mape = mean_absolute_percentage_error(yte, pred)
    r2 = r2_score(yte, pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(reg, X_scaled, y_reg, cv=kf, scoring="r2")

    print(f"[Regression] MAE={mae:.1f} t | MAPE={mape*100:.2f}% | R2={r2:.3f}")
    print(f"[Regression] 5-fold CV R2: {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

    final_reg = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=2,
                                       random_state=42, n_jobs=-1)
    final_reg.fit(X_scaled, y_reg)

    # ---------------- Classification: shortfall probability ----------------
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        X_scaled, y_clf, test_size=0.25, random_state=42, stratify=y_clf)
    clf = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=2,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xtr_c, ytr_c)
    proba = clf.predict_proba(Xte_c)[:, 1]
    pred_c = clf.predict(Xte_c)
    auc = roc_auc_score(yte_c, proba)
    cm = confusion_matrix(yte_c, pred_c).tolist()
    report = classification_report(yte_c, pred_c, output_dict=True)

    cv_auc = cross_val_score(clf, X_scaled, y_clf, cv=kf, scoring="roc_auc")
    print(f"[Classification] ROC-AUC={auc:.3f} | 5-fold CV AUC={cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
    print(confusion_matrix(yte_c, pred_c))

    final_clf = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=2,
                                        class_weight="balanced", random_state=42, n_jobs=-1)
    final_clf.fit(X_scaled, y_clf)

    reg_importances = dict(zip(model_features, final_reg.feature_importances_.tolist()))
    clf_importances = dict(zip(model_features, final_clf.feature_importances_.tolist()))

    # Directional sign of each feature's relationship with the SHORTFALL class,
    # from real correlation in the training data (see prospectivity model for
    # the same rationale — needed so "increases risk" / "reduces risk" in the
    # UI is actually correct rather than assuming every feature is positively
    # correlated with risk).
    direction_signs = {}
    for f in model_features:
        corr = np.corrcoef(df[f].values, y_clf)[0, 1]
        direction_signs[f] = 1.0 if (np.isnan(corr) or corr >= 0) else -1.0

    feature_stats = {
        f: {"mean": float(df[f if f != "mine_type_encoded" else "mine_type_encoded"].mean()),
            "std": float(df[f if f != "mine_type_encoded" else "mine_type_encoded"].std() + 1e-9)}
        for f in model_features
    }

    joblib.dump(final_reg, os.path.join(MODEL_DIR, "production_regressor.joblib"))
    joblib.dump(final_clf, os.path.join(MODEL_DIR, "shortfall_classifier.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "production_scaler.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "mine_type_encoder.joblib"))

    meta = {
        "features": model_features,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "mine_type_classes": le.classes_.tolist(),
        "regression_feature_importances": reg_importances,
        "classification_feature_importances": clf_importances,
        "direction_signs": direction_signs,
        "feature_stats": feature_stats,
        "metrics": {
            "regression": {
                "mae_tonnes": mae, "mape_pct": mape * 100, "r2": r2,
                "cv_r2_mean": float(cv_r2.mean()), "cv_r2_std": float(cv_r2.std()),
                "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            },
            "classification": {
                "roc_auc": auc, "cv_roc_auc_mean": float(cv_auc.mean()),
                "cv_roc_auc_std": float(cv_auc.std()),
                "confusion_matrix": cm, "classification_report": report,
                "n_train": int(len(Xtr_c)), "n_test": int(len(Xte_c)),
            },
        },
        "shortfall_rule": "shortfall_event = 1 if actual_production < 90% of planned_target_tonnes for the month",
        "risk_bands": {"low_max": 0.30, "medium_max": 0.60},
        "model_type": {"regression": "RandomForestRegressor", "classification": "RandomForestClassifier"},
    }
    with open(os.path.join(MODEL_DIR, "production_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved: production_regressor.joblib, shortfall_classifier.joblib, "
          "production_scaler.joblib, mine_type_encoder.joblib, production_meta.json")


if __name__ == "__main__":
    main()
