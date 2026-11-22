"""Rule engine that turns zone analytics into deduplicated, throttled alerts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

SOIL_CRITICAL = 0.12
SOIL_WARNING = 0.18
HEAT_WARNING_C = 36.0
RISK_WARNING = 0.35
RISK_CRITICAL = 0.60
DECLINE_Z = -2.5
DECLINE_SLOPE = -0.01
WETNESS_WINDOW = 0.5


@dataclass(frozen=True)
class Alert:
    zone_id: int
    code: str
    severity: str
    message: str
    created_at: datetime
    evidence: dict = field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        return f"{self.zone_id}:{self.code}"

    @property
    def alert_id(self) -> str:
        stamp = self.created_at.strftime("%Y%m%d%H%M%S")
        return f"{self.dedupe_key}:{stamp}"

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "zone_id": self.zone_id,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "evidence": self.evidence,
        }

    def sms_text(self) -> str:
        return f"[{self.severity.upper()}] Zone {self.zone_id}: {self.message}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def evaluate_zone(row: pd.Series, now: datetime) -> list[Alert]:
    """Apply every rule to a single zone record."""
    alerts: list[Alert] = []
    zone_id = int(row["zone_id"])

    def add(code: str, severity: str, message: str, **evidence: float) -> None:
        alerts.append(
            Alert(
                zone_id=zone_id,
                code=code,
                severity=severity,
                message=message,
                created_at=now,
                evidence={k: float(v) for k, v in evidence.items()},
            )
        )

    risk = float(row.get("risk_probability", row.get("risk_score", 0.0)))
    if risk >= RISK_CRITICAL:
        add("disease_risk", "critical", "High disease risk; scout and treat this zone today.", risk=risk)
    elif risk >= RISK_WARNING:
        add("disease_risk", "warning", "Elevated disease risk; increase scouting frequency.", risk=risk)

    soil = float(row.get("soil_moisture", 1.0))
    if soil < SOIL_CRITICAL:
        add("soil_moisture", "critical", "Soil moisture critically low; irrigate immediately.", soil_moisture=soil)
    elif soil < SOIL_WARNING:
        add("soil_moisture", "warning", "Soil moisture below optimum; schedule irrigation.", soil_moisture=soil)

    temp = float(row.get("air_temp_max", 0.0))
    if temp >= HEAT_WARNING_C:
        add("heat_stress", "warning", "Heat stress window; consider shade and evening irrigation.", air_temp_max=temp)

    z_score = float(row.get("ndvi_z", 0.0))
    slope = float(row.get("ndvi_slope", 0.0))
    if z_score <= DECLINE_Z or slope <= DECLINE_SLOPE:
        add(
            "canopy_decline",
            "warning",
            "Canopy vigour is declining relative to this zone's own history.",
            ndvi_z=z_score,
            ndvi_slope=slope,
        )

    wetness = float(row.get("leaf_wetness_frac", 0.0))
    humidity = float(row.get("humidity_mean", 0.0))
    if wetness >= WETNESS_WINDOW and humidity >= 85.0:
        add(
            "leaf_wetness_window",
            "info",
            "Extended leaf wetness; fungal infection window is open.",
            leaf_wetness_frac=wetness,
            humidity_mean=humidity,
        )
    return alerts


def evaluate_zones(zones: pd.DataFrame, now: datetime | None = None) -> list[Alert]:
    now = now or _utcnow()
    alerts: list[Alert] = []
    for _, row in zones.iterrows():
        alerts.extend(evaluate_zone(row, now))
    return alerts


class AlertEngine:
    """Deduplicates and rate-limits alerts per zone and rule code."""

    def __init__(self, cooldown_hours: float = 6.0, clock: Callable[[], datetime] = _utcnow):
        self.cooldown = timedelta(hours=cooldown_hours)
        self._clock = clock
        self._last_sent: dict[str, datetime] = {}
        self.sent: list[Alert] = []

    def filter(self, alerts: Iterable[Alert]) -> list[Alert]:
        now = self._clock()
        kept: list[Alert] = []
        for alert in alerts:
            last = self._last_sent.get(alert.dedupe_key)
            if last is not None and now - last < self.cooldown:
                continue
            self._last_sent[alert.dedupe_key] = now
            kept.append(alert)
        return sorted(kept, key=lambda a: (-SEVERITY_ORDER[a.severity], a.zone_id))

    def dispatch(self, alerts: Iterable[Alert], sender: Callable[[Alert], bool] | None = None) -> list[Alert]:
        """Record and optionally deliver alerts. ``sender`` returns success."""
        accepted = self.filter(alerts)
        for alert in accepted:
            if sender is not None:
                sender(alert)
            self.sent.append(alert)
        return accepted

    def history_frame(self) -> pd.DataFrame:
        if not self.sent:
            return pd.DataFrame(columns=["alert_id", "zone_id", "code", "severity", "message", "created_at"])
        return pd.DataFrame([alert.to_dict() for alert in self.sent])
