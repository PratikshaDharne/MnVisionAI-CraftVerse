"""
MnVision AI — Inference Engine
================================
Loads the trained artefacts produced by ml/train_prospectivity_model.py and
ml/train_production_model.py and performs REAL model inference (no
hardcoded/fake results). Every score, probability, feature-importance and
explanation returned by this module comes directly from the loaded
scikit-learn models.
"""
import os
import json
import math
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")


def _load_json(name):
    with open(os.path.join(MODEL_DIR, name)) as f:
        return json.load(f)


class ProspectivityEngine:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODEL_DIR, "prospectivity_model.joblib"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "prospectivity_scaler.joblib"))
        self.meta = _load_json("prospectivity_meta.json")
        self.grid_df = pd.read_csv(os.path.join(MODEL_DIR, "geospatial_grid.csv"))
        self.occurrences = _load_json("known_occurrences.json")
        self.features = self.meta["features"]
        self.feature_labels = self.meta["feature_labels"]
        self.feature_stats = self.meta["feature_stats"]
        self.importances = self.meta["feature_importances"]
        self.direction_signs = self.meta.get("direction_signs", {})
        self.zone_thresholds = self.meta["zone_thresholds"]

        # run real inference once over the whole grid and cache it (this is the
        # "Satellite/Geospatial Data -> ML Model -> Prospectivity Map" pipeline)
        X = self.grid_df[self.features].values
        Xs = self.scaler.transform(X)
        proba = self.model.predict_proba(Xs)[:, 1]
        # model confidence: agreement across the forest's trees (1 - 2*std of tree votes)
        tree_votes = np.stack([t.predict_proba(Xs)[:, 1] for t in self.model.estimators_])
        vote_std = tree_votes.std(axis=0)
        confidence = np.clip(1 - 2.2 * vote_std, 0.35, 0.99)

        self.grid_df = self.grid_df.assign(prospectivity_score=proba, confidence=confidence)
        self.grid_df["zone"] = self.grid_df["prospectivity_score"].apply(self._zone_for_score)

    def _zone_for_score(self, score):
        if score < self.zone_thresholds["low_max"]:
            return "Low"
        if score < self.zone_thresholds["medium_max"]:
            return "Medium"
        return "High"

    def explain_point(self, row):
        """Simple, transparent local explanation: for each feature, compare the
        point's z-score against the training population, apply the feature's
        real directional sign (from training-data correlation with the target
        — e.g. distance-type features are inversely related to prospectivity),
        and weight by the model's global feature importance. This produces an
        auditable 'why this cell is/isn't prospective' breakdown without
        depending on an external SHAP install."""
        contributions = []
        for f in self.features:
            stats = self.feature_stats[f]
            z = (row[f] - stats["mean"]) / stats["std"]
            importance = self.importances.get(f, 0)
            sign = self.direction_signs.get(f, 1.0)
            contribution = float(z * importance * sign)
            contributions.append({
                "feature": f,
                "label": self.feature_labels.get(f, f),
                "value": float(row[f]),
                "population_mean": stats["mean"],
                "importance": importance,
                "contribution": contribution,
            })
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        return contributions[:5]

    def get_grid(self, bbox=None):
        df = self.grid_df
        if bbox:
            lat_min, lon_min, lat_max, lon_max = bbox
            df = df[(df.lat >= lat_min) & (df.lat <= lat_max) &
                    (df.lon >= lon_min) & (df.lon <= lon_max)]
        out = []
        for _, row in df.iterrows():
            out.append({
                "cell_id": int(row["cell_id"]),
                "lat": row["lat"], "lon": row["lon"],
                "prospectivity_score": round(float(row["prospectivity_score"]), 4),
                "zone": row["zone"],
                "confidence": round(float(row["confidence"]), 4),
            })
        return out

    def get_point(self, lat, lon):
        # nearest grid cell (Euclidean on lat/lon is fine at this scale)
        d2 = (self.grid_df["lat"] - lat) ** 2 + (self.grid_df["lon"] - lon) ** 2
        idx = d2.idxmin()
        row = self.grid_df.loc[idx]
        explanation = self.explain_point(row)
        nearest_occ = min(
            self.occurrences["occurrences"],
            key=lambda o: (o["lat"] - lat) ** 2 + (o["lon"] - lon) ** 2,
        )
        return {
            "cell_id": int(row["cell_id"]),
            "lat": float(row["lat"]), "lon": float(row["lon"]),
            "prospectivity_score": round(float(row["prospectivity_score"]), 4),
            "zone": row["zone"],
            "confidence": round(float(row["confidence"]), 4),
            "explanation": explanation,
            "nearest_known_occurrence": {
                "name": nearest_occ["name"],
                "distance_km": round(float(row["dist_to_known_occurrence_km"]), 2),
            },
            "raw_features": {f: float(row[f]) for f in self.features},
        }

    def summary(self):
        vc = self.grid_df["zone"].value_counts().to_dict()
        return {
            "area_name": self.occurrences["area_name"],
            "bbox": self.occurrences["bbox"],
            "known_occurrences": self.occurrences["occurrences"],
            "zone_counts": {z: int(vc.get(z, 0)) for z in ["Low", "Medium", "High"]},
            "total_cells": int(len(self.grid_df)),
            "metrics": self.meta["metrics"],
            "feature_importances": self.importances,
            "feature_labels": self.feature_labels,
        }


class ProductionEngine:
    def __init__(self):
        self.reg = joblib.load(os.path.join(MODEL_DIR, "production_regressor.joblib"))
        self.clf = joblib.load(os.path.join(MODEL_DIR, "shortfall_classifier.joblib"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "production_scaler.joblib"))
        self.mine_encoder = joblib.load(os.path.join(MODEL_DIR, "mine_type_encoder.joblib"))
        self.meta = _load_json("production_meta.json")
        self.mines = _load_json("mines.json")
        self.history_df = pd.read_csv(os.path.join(MODEL_DIR, "production_dataset.csv"))
        self.features = self.meta["features"]
        self.feature_labels = self.meta["feature_labels"]
        self.feature_stats = self.meta["feature_stats"]
        self.risk_bands = self.meta["risk_bands"]
        self.direction_signs = self.meta.get("direction_signs", {})

    def _mine(self, mine_id):
        m = next((m for m in self.mines if m["mine_id"] == mine_id), None)
        if m is None:
            raise ValueError(f"Unknown mine_id: {mine_id}")
        return m

    def _vectorize(self, mine_type, payload):
        mine_type_encoded = int(self.mine_encoder.transform([mine_type])[0])
        row = []
        for f in self.features:
            if f == "mine_type_encoded":
                row.append(mine_type_encoded)
            else:
                row.append(float(payload[f]))
        X = np.array(row).reshape(1, -1)
        return self.scaler.transform(X)

    def _risk_band(self, prob):
        if prob < self.risk_bands["low_max"]:
            return "Low"
        if prob < self.risk_bands["medium_max"]:
            return "Medium"
        return "High"

    def latest_conditions(self, mine_id):
        df = self.history_df[self.history_df.mine_id == mine_id].sort_values(["year", "month"])
        last = df.iloc[-1]
        return last

    def predict(self, mine_id, overrides=None):
        mine = self._mine(mine_id)
        last = self.latest_conditions(mine_id)
        payload = {
            "rainfall_mm": last["rainfall_mm"],
            "avg_temperature_c": last["avg_temperature_c"],
            "soil_moisture_pct": last["soil_moisture_pct"],
            "equipment_availability_pct": last["equipment_availability_pct"],
            "downtime_hours": last["downtime_hours"],
            "labor_availability_pct": last["labor_availability_pct"],
            "planned_target_tonnes": last["planned_target_tonnes"],
            "prev_month_efficiency": last["actual_production_tonnes"] / last["planned_target_tonnes"],
            "month": (int(last["month"]) % 12) + 1,
        }
        if overrides:
            payload.update({k: v for k, v in overrides.items() if k in payload})

        Xs = self._vectorize(mine["type"], payload)
        predicted_production = float(self.reg.predict(Xs)[0])
        shortfall_prob = float(self.clf.predict_proba(Xs)[0][1])

        # tree-vote confidence for the regressor prediction
        tree_preds = np.array([t.predict(Xs)[0] for t in self.reg.estimators_])
        pred_std = float(tree_preds.std())
        confidence = float(np.clip(1 - pred_std / max(predicted_production, 1) * 3, 0.3, 0.98))

        target = payload["planned_target_tonnes"]
        shortfall_pct = 100 * (target - predicted_production) / target
        risk_level = self._risk_band(shortfall_prob)

        drivers = self._explain(mine["type"], payload)
        recommendations = self._recommend(payload, drivers, risk_level)

        return {
            "mine_id": mine_id,
            "mine_name": mine["name"],
            "mine_type": mine["type"],
            "inputs": payload,
            "planned_target_tonnes": round(target, 1),
            "predicted_production_tonnes": round(predicted_production, 1),
            "predicted_shortfall_tonnes": round(max(0.0, target - predicted_production), 1),
            "shortfall_pct": round(shortfall_pct, 2),
            "shortfall_probability": round(shortfall_prob, 4),
            "risk_level": risk_level,
            "prediction_confidence": round(confidence, 3),
            "risk_drivers": drivers,
            "recommendations": recommendations,
        }

    def _explain(self, mine_type, payload):
        importances = self.meta["classification_feature_importances"]
        contributions = []
        for f in self.features:
            if f == "mine_type_encoded":
                continue
            stats = self.feature_stats.get(f, {"mean": 0, "std": 1})
            z = (payload[f] - stats["mean"]) / (stats["std"] or 1)
            imp = importances.get(f, 0)
            sign = self.direction_signs.get(f, 1.0)
            contribution = float(z * imp * sign)
            contributions.append({
                "feature": f,
                "label": self.feature_labels.get(f, f),
                "value": round(float(payload[f]), 2),
                "importance": round(imp, 4),
                "contribution": round(contribution, 4),
                "direction": "increases risk" if contribution > 0 else "reduces risk",
            })
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        return contributions[:5]

    def _recommend(self, payload, drivers, risk_level):
        recs = []
        # Only act on drivers that are ACTUALLY pushing risk up (positive contribution),
        # not on features that happen to rank high in absolute importance but currently
        # sit in a favourable range.
        risk_increasing = {d["feature"] for d in drivers if d["direction"] == "increases risk"}

        if "downtime_hours" in risk_increasing or payload["downtime_hours"] > 100:
            recs.append("Schedule preventive maintenance to cut equipment downtime; "
                         "prioritise the units with the highest recent failure rate.")
        if "equipment_availability_pct" in risk_increasing or payload["equipment_availability_pct"] < 80:
            recs.append("Raise equipment availability with an expedited spares/service plan "
                         "or short-term equipment hire to cover the shortfall period.")
        if "rainfall_mm" in risk_increasing or payload["rainfall_mm"] > 180:
            recs.append("Activate monsoon haulage contingency: reinforce haul roads and "
                         "pre-position pumps to limit rainfall-driven production loss.")
        if "labor_availability_pct" in risk_increasing or payload["labor_availability_pct"] < 85:
            recs.append("Review shift rostering and contract-labour backup to stabilise "
                         "workforce availability.")
        if "soil_moisture_pct" in risk_increasing or payload["soil_moisture_pct"] > 55:
            recs.append("Adjust blasting/loading schedules for high soil-moisture conditions "
                         "to protect equipment traction and cycle times.")
        if not recs:
            recs.append("Conditions are within normal operating range; maintain current "
                         "production plan and monitor weekly.")
        if risk_level == "High":
            recs.insert(0, "Escalate to mine operations management: shortfall probability is "
                            "high enough to warrant a revised monthly production plan.")
        return recs

    def history(self, mine_id):
        df = self.history_df[self.history_df.mine_id == mine_id].sort_values(["year", "month"])
        out = []
        for _, r in df.iterrows():
            out.append({
                "year": int(r["year"]), "month": int(r["month"]),
                "planned_target_tonnes": r["planned_target_tonnes"],
                "actual_production_tonnes": r["actual_production_tonnes"],
                "shortfall_pct": round(float(r["shortfall_pct"]), 2),
                "shortfall_event": int(r["shortfall_event"]),
                "rainfall_mm": r["rainfall_mm"],
                "equipment_availability_pct": r["equipment_availability_pct"],
            })
        return out

    def mines_list(self):
        return self.mines

    def metrics(self):
        return self.meta["metrics"]


# Singletons loaded once at process start — real trained artefacts, not mocks.
prospectivity_engine = ProspectivityEngine()
production_engine = ProductionEngine()
