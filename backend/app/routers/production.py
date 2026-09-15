from fastapi import APIRouter, HTTPException
from ..inference import production_engine as engine
from ..schemas import WhatIfRequest

router = APIRouter(prefix="/api/production", tags=["production"])


@router.get("/mines")
def list_mines():
    return engine.mines_list()


@router.get("/history")
def history(mine_id: str):
    try:
        return {"mine_id": mine_id, "records": engine.history(mine_id)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/predict")
def predict(mine_id: str):
    try:
        return engine.predict(mine_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/whatif")
def whatif(req: WhatIfRequest):
    overrides = req.dict(exclude={"mine_id"}, exclude_none=True)
    try:
        return engine.predict(req.mine_id, overrides=overrides)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/metrics")
def metrics():
    return engine.metrics()
