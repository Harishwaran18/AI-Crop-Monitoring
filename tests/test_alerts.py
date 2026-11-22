from datetime import datetime, timedelta, timezone

import pandas as pd

from cropmon.alerts.rules import Alert, AlertEngine, evaluate_zone, evaluate_zones

NOW = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)


def _row(**overrides):
    base = {
        "zone_id": 3,
        "risk_probability": 0.1,
        "soil_moisture": 0.30,
        "air_temp_max": 28.0,
        "ndvi_z": 0.0,
        "ndvi_slope": 0.0,
        "leaf_wetness_frac": 0.1,
        "humidity_mean": 60.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_high_risk_produces_critical_alert():
    alerts = evaluate_zone(_row(risk_probability=0.85), NOW)
    codes = {(a.code, a.severity) for a in alerts}
    assert ("disease_risk", "critical") in codes


def test_moderate_risk_produces_warning():
    alerts = evaluate_zone(_row(risk_probability=0.45), NOW)
    assert any(a.code == "disease_risk" and a.severity == "warning" for a in alerts)


def test_healthy_zone_has_no_alerts():
    assert evaluate_zone(_row(), NOW) == []


def test_dry_soil_and_heat_rules():
    alerts = evaluate_zone(_row(soil_moisture=0.10, air_temp_max=39.0), NOW)
    codes = {a.code for a in alerts}
    assert "soil_moisture" in codes
    assert "heat_stress" in codes
    critical = [a for a in alerts if a.code == "soil_moisture"][0]
    assert critical.severity == "critical"


def test_canopy_decline_uses_z_score_or_slope():
    assert any(a.code == "canopy_decline" for a in evaluate_zone(_row(ndvi_z=-3.0), NOW))
    assert any(a.code == "canopy_decline" for a in evaluate_zone(_row(ndvi_slope=-0.02), NOW))


def test_leaf_wetness_window():
    alerts = evaluate_zone(_row(leaf_wetness_frac=0.8, humidity_mean=92.0), NOW)
    assert any(a.code == "leaf_wetness_window" for a in alerts)


def test_evaluate_zones_iterates_frame():
    frame = pd.DataFrame([_row(zone_id=1, soil_moisture=0.15), _row(zone_id=2, risk_probability=0.9)])
    alerts = evaluate_zones(frame, NOW)
    assert {a.zone_id for a in alerts} == {1, 2}


def test_alert_serialisation():
    alert = evaluate_zone(_row(risk_probability=0.9), NOW)[0]
    payload = alert.to_dict()
    assert payload["created_at"].startswith("2025-07-01T12:00")
    assert payload["alert_id"].startswith("3:disease_risk")
    assert "Zone 3" in alert.sms_text()


def test_engine_deduplicates_within_cooldown():
    now = [NOW]

    def clock():
        return now[0]

    engine = AlertEngine(cooldown_hours=6, clock=clock)
    alert = evaluate_zone(_row(risk_probability=0.9), NOW)[0]
    assert len(engine.filter([alert])) == 1
    assert engine.filter([alert]) == []

    now[0] = NOW + timedelta(hours=7)
    assert len(engine.filter([alert])) == 1


def test_engine_sorts_by_severity():
    engine = AlertEngine(cooldown_hours=0)
    frame = pd.DataFrame(
        [
            _row(zone_id=1, leaf_wetness_frac=0.9, humidity_mean=95.0),
            _row(zone_id=2, risk_probability=0.95, soil_moisture=0.05),
        ]
    )
    accepted = engine.filter(evaluate_zones(frame, NOW))
    assert accepted[0].severity == "critical"


def test_engine_dispatch_calls_sender():
    engine = AlertEngine(cooldown_hours=0)
    sent = []
    alerts = [Alert(zone_id=1, code="test", severity="info", message="m", created_at=NOW)]
    engine.dispatch(alerts, sender=lambda alert: sent.append(alert) or True)
    assert len(sent) == 1
    assert len(engine.history_frame()) == 1


def test_history_frame_empty():
    assert AlertEngine().history_frame().empty
