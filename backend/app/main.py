from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import prospectivity, production, alerts, report

app = FastAPI(
    title="MnVision AI — Manganese Mining Intelligence API",
    description=("Backend serving REAL ML inference for manganese prospectivity mapping "
                 "and production shortfall prediction over the Nagpur–Bhandara–Balaghat "
                 "manganese belt (SIH26009 prototype)."),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prospectivity.router)
app.include_router(production.router)
app.include_router(alerts.router)
app.include_router(report.router)


@app.get("/")
def root():
    return {
        "service": "MnVision AI Backend",
        "status": "online",
        "docs": "/docs",
        "endpoints": [
            "/api/prospectivity/summary", "/api/prospectivity/grid",
            "/api/prospectivity/point?lat=..&lon=..", "/api/prospectivity/occurrences",
            "/api/production/mines", "/api/production/history?mine_id=..",
            "/api/production/predict?mine_id=..", "/api/production/whatif",
            "/api/production/metrics", "/api/alerts", "/api/report",
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
