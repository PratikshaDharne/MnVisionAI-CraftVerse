from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from ..inference import prospectivity_engine as engine

router = APIRouter(prefix="/api/prospectivity", tags=["prospectivity"])


@router.get("/summary")
def get_summary():
    return engine.summary()


@router.get("/grid")
def get_grid(lat_min: Optional[float] = Query(None), lon_min: Optional[float] = Query(None),
             lat_max: Optional[float] = Query(None), lon_max: Optional[float] = Query(None)):
    bbox = None
    if None not in (lat_min, lon_min, lat_max, lon_max):
        bbox = (lat_min, lon_min, lat_max, lon_max)
    cells = engine.get_grid(bbox)
    return {"count": len(cells), "cells": cells}


@router.get("/point")
def get_point(lat: float = Query(...), lon: float = Query(...)):
    if not (engine.occurrences["bbox"][0] - 1 <= lat <= engine.occurrences["bbox"][2] + 1):
        raise HTTPException(status_code=400, detail="Latitude outside supported study area")
    return engine.get_point(lat, lon)


@router.get("/occurrences")
def get_occurrences():
    return engine.occurrences
