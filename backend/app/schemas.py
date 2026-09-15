from typing import Optional, Dict, List
from pydantic import BaseModel


class PointQuery(BaseModel):
    lat: float
    lon: float


class WhatIfRequest(BaseModel):
    mine_id: str
    rainfall_mm: Optional[float] = None
    avg_temperature_c: Optional[float] = None
    soil_moisture_pct: Optional[float] = None
    equipment_availability_pct: Optional[float] = None
    downtime_hours: Optional[float] = None
    labor_availability_pct: Optional[float] = None
    planned_target_tonnes: Optional[float] = None
