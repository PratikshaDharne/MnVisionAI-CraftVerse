"""
Generates a plain-text/markdown decision report assembled entirely from live
model outputs (prospectivity summary + per-mine production predictions).
No canned text describing results — only the surrounding narrative template
is fixed, all figures are pulled from the engines at request time.
"""
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from ..inference import prospectivity_engine, production_engine

router = APIRouter(prefix="/api/report", tags=["report"])


@router.get("", response_class=PlainTextResponse)
def generate_report():
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    summary = prospectivity_engine.summary()

    lines = []
    lines.append("=" * 72)
    lines.append("MnVision AI — Decision Support Report")
    lines.append(f"Area: {summary['area_name']}")
    lines.append(f"Generated: {ts}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. MANGANESE PROSPECTIVITY SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total grid cells analysed: {summary['total_cells']}")
    for zone in ["High", "Medium", "Low"]:
        lines.append(f"  {zone} prospectivity zone cells: {summary['zone_counts'][zone]}")
    m = summary["metrics"]
    lines.append(f"Model validation — ROC-AUC: {m['test_roc_auc']:.3f} "
                 f"(5-fold CV: {m['cv_roc_auc_mean']:.3f} +/- {m['cv_roc_auc_std']:.3f}), "
                 f"Average Precision: {m['test_average_precision']:.3f}")
    lines.append("Top geological/geospatial drivers of prospectivity (global model importance):")
    ranked = sorted(summary["feature_importances"].items(), key=lambda x: -x[1])
    for f, imp in ranked[:5]:
        lines.append(f"  - {summary['feature_labels'].get(f, f)}: {imp*100:.1f}%")
    lines.append("")
    lines.append("Known documented occurrences anchoring the model (presence/background labels):")
    for occ in summary["known_occurrences"]:
        lines.append(f"  - {occ['name']} ({occ['lat']:.3f}, {occ['lon']:.3f})")
    lines.append("")
    lines.append("Note: Prospectivity scores are an exploration-prioritisation ranking derived")
    lines.append("from surface geological/geospatial proxies. They do NOT directly detect")
    lines.append("underground ore and must be confirmed via field validation / drilling.")
    lines.append("")

    lines.append("2. PRODUCTION & SHORTFALL RISK SUMMARY")
    lines.append("-" * 40)
    for mine in production_engine.mines_list():
        pred = production_engine.predict(mine["mine_id"])
        lines.append(f"Mine: {pred['mine_name']} ({pred['mine_id']}, {pred['mine_type']})")
        lines.append(f"  Planned target: {pred['planned_target_tonnes']:.0f} t | "
                     f"Predicted production: {pred['predicted_production_tonnes']:.0f} t | "
                     f"Predicted shortfall: {pred['predicted_shortfall_tonnes']:.0f} t "
                     f"({pred['shortfall_pct']:.1f}%)")
        lines.append(f"  Shortfall probability: {pred['shortfall_probability']*100:.1f}% "
                     f"-> Risk level: {pred['risk_level']} "
                     f"(prediction confidence {pred['prediction_confidence']*100:.0f}%)")
        lines.append("  Top risk drivers:")
        for d in pred["risk_drivers"][:3]:
            lines.append(f"    - {d['label']}: value={d['value']} ({d['direction']})")
        lines.append("  Recommendations:")
        for r in pred["recommendations"]:
            lines.append(f"    - {r}")
        lines.append("")

    rm = production_engine.metrics()
    lines.append("Model validation — Production regressor R2: "
                 f"{rm['regression']['r2']:.3f} (CV {rm['regression']['cv_r2_mean']:.3f}), "
                 f"MAPE: {rm['regression']['mape_pct']:.1f}% | "
                 f"Shortfall classifier ROC-AUC: {rm['classification']['roc_auc']:.3f} "
                 f"(CV {rm['classification']['cv_roc_auc_mean']:.3f})")
    lines.append("")
    lines.append("=" * 72)
    lines.append("End of report — generated directly from live MnVision AI model inference.")
    return "\n".join(lines)
