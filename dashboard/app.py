"""Unified Streamlit dashboard for the crop monitoring platform.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropmon.alerts.channels import (  # noqa: E402
    ChannelNotConfigured,
    RateLimiter,
    build_channel,
    validate_e164,
)
from cropmon.alerts.rules import Alert  # noqa: E402
from cropmon.config import Settings  # noqa: E402
from cropmon.features.temporal import detect_anomalies  # noqa: E402
from cropmon.pipeline import load_artifacts, run_pipeline  # noqa: E402

st.set_page_config(page_title="Crop Monitor | SIH25099", page_icon="🌱", layout="wide")

SEVERITY_COLORS = {"critical": "#b3001b", "warning": "#d98324", "info": "#3d7ea6"}


@st.cache_resource(show_spinner="Running analysis pipeline (first load only)...")
def get_artifacts() -> dict:
    settings = Settings.from_env()
    try:
        return load_artifacts(settings)
    except FileNotFoundError:
        run_pipeline(settings)
        return load_artifacts(settings)


@st.cache_data
def load_maps(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def render_overview(art: dict) -> None:
    zones: pd.DataFrame = art["zones"]
    summary: dict = art["summary"]
    maps = art["maps"]

    columns = st.columns(4)
    columns[0].metric("Mean canopy health", f"{summary['mean_health_score']:.1f}/100")
    columns[1].metric("Mean NDVI", f"{summary['mean_ndvi']:.3f}")
    columns[2].metric("High-risk zones", f"{summary['risk_level_counts'].get('high', 0)} / {summary['n_zones']}")
    columns[3].metric("Alerts raised", summary["alert_count"])

    left, right = st.columns([1.35, 1])
    with left:
        figure = px.imshow(
            maps["risk"],
            color_continuous_scale="YlOrRd",
            zmin=0.0,
            zmax=1.0,
            labels={"color": "risk probability"},
            title="Predicted disease risk by zone",
        )
        figure.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = px.imshow(
            maps["ndvi"],
            color_continuous_scale="RdYlGn",
            zmin=0.1,
            zmax=0.9,
            labels={"color": "NDVI"},
            title="Canopy vigour (NDVI)",
        )
        figure.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Zones needing attention")
    priority = zones.sort_values("risk_probability", ascending=False).head(12)
    display = priority[
        ["zone_id", "risk_probability", "risk_level", "health_score", "health_class", "ndvi", "soil_moisture"]
    ].copy()
    display["risk_probability"] = display["risk_probability"].round(2)
    display["ndvi"] = display["ndvi"].round(3)
    display["soil_moisture"] = display["soil_moisture"].round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.info(
        "Zones combine spectral indices, weekly canopy trend, and micro-climate. "
        "All values on this page come from the built-in simulator, not field hardware."
    )


def render_spectral(art: dict) -> None:
    maps = art["maps"]
    zones: pd.DataFrame = art["zones"]
    index_names = art["summary"].get("index_names", ["ndvi"])

    left, right = st.columns([1, 2.2])
    with left:
        default_index = index_names.index("ndvi") if "ndvi" in index_names else 0
        selected = st.selectbox("Index", index_names, index=default_index)
        st.caption(art["index_descriptions"].get(selected, ""))
        zone_id = st.selectbox("Inspect zone", sorted(zones["zone_id"].tolist()))
        row = zones.loc[zones["zone_id"] == zone_id].iloc[0]
        st.metric("Zone health", f"{row['health_score']:.0f}/100", row["health_class"])
        st.metric("Risk probability", f"{row['risk_probability']:.2f}", row["risk_level"])

    with right:
        figure = px.imshow(
            maps[selected],
            color_continuous_scale="Viridis",
            labels={"color": selected.upper()},
            title=f"{selected.upper()} map",
        )
        figure.update_layout(height=470, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Zone signature vs field average")
    index_frame = zones.set_index("zone_id")[index_names]
    field_mean = index_frame.mean()
    zone_values = index_frame.loc[zone_id]
    comparison = pd.DataFrame({"zone": zone_values, "field_mean": field_mean}).reset_index(names="index")
    figure = go.Figure()
    figure.add_bar(x=comparison["index"], y=comparison["zone"], name=f"zone {zone_id}", marker_color="#2e8b57")
    figure.add_bar(x=comparison["index"], y=comparison["field_mean"], name="field mean", marker_color="#b0b0b0")
    figure.update_layout(barmode="group", height=340, yaxis_title="index value", margin=dict(t=20))
    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "NDVI/GNDVI/NDRE track chlorophyll and canopy density, SAVI corrects for soil background, "
        "NDWI tracks canopy water, PSRI detects senescence, CIre is sensitive to chlorophyll at high leaf area."
    )


def render_temporal(art: dict) -> None:
    history: pd.DataFrame = art["history"]
    sensors: pd.DataFrame = art["sensors"]
    zones: pd.DataFrame = art["zones"]

    pivot = history.pivot(index="week", columns="zone_id", values="ndvi")
    field_mean = pivot.mean(axis=1)
    mask, _ = detect_anomalies(field_mean.to_numpy(), z_threshold=2.0)

    figure = go.Figure()
    figure.add_scatter(
        x=pivot.index, y=field_mean, mode="lines+markers", name="field mean NDVI", line=dict(color="#2e8b57")
    )
    figure.add_scatter(
        x=pivot.index[mask],
        y=field_mean[mask],
        mode="markers",
        name="anomaly",
        marker=dict(color="#b3001b", size=12, symbol="x"),
    )
    figure.update_layout(height=340, title="Field mean NDVI with anomaly detection", yaxis_title="NDVI")
    st.plotly_chart(figure, use_container_width=True)

    worst = zones.sort_values("risk_probability", ascending=False)["zone_id"].head(4).tolist()
    selected = st.multiselect("Zones to overlay", pivot.columns.tolist(), default=worst)
    if selected:
        overlay = go.Figure()
        for zone in selected:
            overlay.add_scatter(x=pivot.index, y=pivot[zone], mode="lines+markers", name=f"zone {zone}")
        overlay.update_layout(height=320, yaxis_title="NDVI", title="Per-zone canopy trajectory")
        st.plotly_chart(overlay, use_container_width=True)

    st.subheader("Sensor history")
    daily = (
        sensors.set_index("timestamp")
        .resample("1D")
        .agg(
            air_temp_c=("air_temp_c", "mean"),
            humidity_pct=("humidity_pct", "mean"),
            soil_moisture=("soil_moisture", "mean"),
            rainfall_mm=("rainfall_mm", "sum"),
        )
        .reset_index()
    )
    tabs = st.tabs(["Temperature & humidity", "Soil moisture", "Rainfall"])
    with tabs[0]:
        figure = go.Figure()
        figure.add_scatter(x=daily["timestamp"], y=daily["air_temp_c"], name="air temp (C)", line=dict(color="#c0392b"))
        figure.add_scatter(
            x=daily["timestamp"], y=daily["humidity_pct"], name="humidity (%)", yaxis="y2", line=dict(color="#2c7fb8")
        )
        figure.update_layout(
            height=340,
            yaxis=dict(title="C"),
            yaxis2=dict(title="%", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(figure, use_container_width=True)
    with tabs[1]:
        figure = px.line(daily, x="timestamp", y="soil_moisture", markers=True)
        figure.add_hline(y=0.18, line_dash="dash", line_color="#d98324", annotation_text="irrigation trigger")
        figure.update_layout(height=320)
        st.plotly_chart(figure, use_container_width=True)
    with tabs[2]:
        figure = px.bar(daily, x="timestamp", y="rainfall_mm")
        figure.update_layout(height=320)
        st.plotly_chart(figure, use_container_width=True)


def _alerts_as_objects(alerts: list[dict]) -> list[Alert]:
    return [
        Alert(
            zone_id=item["zone_id"],
            code=item["code"],
            severity=item["severity"],
            message=item["message"],
            created_at=datetime.fromisoformat(item["created_at"]),
            evidence=item.get("evidence", {}),
        )
        for item in alerts
    ]


def render_alerts(art: dict) -> None:
    alerts = art["alerts"]
    if not alerts:
        st.success("No alerts raised for the current snapshot.")
        return

    frame = pd.DataFrame(alerts)
    severities = st.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning", "info"])
    codes = st.multiselect("Rule", sorted(frame["code"].unique()), default=sorted(frame["code"].unique()))
    filtered = frame[frame["severity"].isin(severities) & frame["code"].isin(codes)]

    counts = st.columns(3)
    for column, severity in zip(counts, ["critical", "warning", "info"], strict=True):
        column.metric(severity.title(), int((frame["severity"] == severity).sum()))

    styled = filtered[["zone_id", "severity", "code", "message", "created_at"]].copy()
    st.dataframe(
        styled.style.map(lambda value: f"color: {SEVERITY_COLORS.get(value, 'inherit')}", subset=["severity"]),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download alerts (CSV)",
        filtered.to_csv(index=False).encode(),
        file_name="crop_alerts.csv",
        mime="text/csv",
    )

    st.subheader("Send SMS for critical alerts")
    st.caption(
        "Credentials are read from the environment (TWILIO_* or FAST2SMS_API_KEY + ALERT_RECIPIENT). "
        "With no credentials the console channel prints the message instead of sending it."
    )
    channel_name = st.selectbox("Channel", ["console", "twilio", "fast2sms"])
    recipient = st.text_input("Recipient (E.164, optional if ALERT_RECIPIENT is set)", value="")
    max_send = st.slider("Maximum messages this session", 1, 5, 2)

    if st.button("Send critical alerts", type="primary"):
        critical = [alert for alert in _alerts_as_objects(alerts) if alert.severity == "critical"]
        if not critical:
            st.warning("No critical alerts to send.")
            return
        try:
            channel = build_channel(channel_name)
            target = validate_e164(recipient) if recipient else None
        except (ChannelNotConfigured, ValueError) as exc:
            st.error(f"Cannot send: {exc}")
            return
        limiter = RateLimiter(max_per_hour=max_send)
        for alert in critical[:max_send]:
            if not limiter.allow():
                st.warning("Session message cap reached.")
                break
            result = channel.send(alert.sms_text(), to=target)
            st.write(f"Zone {alert.zone_id}: {'sent' if result.ok else 'failed'} ({result.detail})")
        st.caption(f"{max(len(critical) - max_send, 0)} critical alerts not sent by the session cap.")

    with st.expander("Raw alert payload"):
        st.json(alerts[:5])


def render_model(art: dict) -> None:
    metrics = art["metrics"]
    summary = art["summary"]

    columns = st.columns(4)
    columns[0].metric("Holdout accuracy", f"{metrics['accuracy']:.3f}")
    columns[1].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    columns[2].metric("ROC AUC (OvR)", f"{metrics['roc_auc_ovr']:.3f}")
    columns[3].metric("Training zones", metrics["n_train"])

    left, right = st.columns(2)
    with left:
        matrix = np.asarray(metrics["confusion_matrix"])
        labels = ["healthy", "moderate", "high"][: matrix.shape[0]]
        figure = px.imshow(
            matrix,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="predicted", y="actual", color="zones"),
            title="Holdout confusion matrix",
        )
        figure.update_layout(height=380, margin=dict(t=50, b=20))
        st.plotly_chart(figure, use_container_width=True)
    with right:
        importance = pd.DataFrame(metrics["feature_importance"]).head(12).iloc[::-1]
        figure = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Permutation importance",
            color="importance",
            color_continuous_scale="YlGn",
        )
        figure.update_layout(height=380, margin=dict(t=50, b=20), coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True)

    with st.expander("Model card and limitations", expanded=True):
        st.markdown(
            f"""
**Algorithm** gradient-boosted trees over {len(metrics['feature_importance'])} fused features
(7 spectral indices, 3 temporal features, 7 micro-climate aggregates).

**Training data** {summary['data_source']}. Metrics measure how well the model
reproduces the simulator's agronomic risk model; they are not a field-validated
performance claim.

**Intended use** triage and scouting prioritisation. Every alert should be
confirmed by a human scout before treatment decisions.

**Known limitations** no real imagery, no pest counts, no soil texture, no
calibration against yield. Replace `cropmon.data.synthetic` with an ingest
adapter for Sentinel-2/Landsat and field sensors before operational use.
"""
        )
        st.json({"model": art["model"].metadata(), "artifact_version": summary["artifact_version"]})


def main() -> None:
    art = get_artifacts()
    settings: Settings = art["settings"]
    art["maps"] = load_maps(str(settings.data_dir / "spectral_maps.npz"))

    st.title("🌱 AI Crop Monitoring")
    st.caption(
        "SIH25099 · multispectral indices + micro-climate fusion · zone risk scoring · SMS alerting"
    )

    with st.sidebar:
        st.header("Snapshot")
        summary = art["summary"]
        st.write(f"Generated: {summary['generated_at'][:19].replace('T', ' ')} UTC")
        st.write(f"Field: {summary['field_shape'][0]}×{summary['field_shape'][1]} px")
        st.write(f"Zones: {summary['n_zones']}")
        st.write(f"Seed: {settings.seed}")
        st.divider()
        st.write("Risk mix")
        for level in ["low", "moderate", "high"]:
            st.write(f"{level.title()}: {summary['risk_level_counts'].get(level, 0)}")
        st.divider()
        if st.button("Recompute snapshot"):
            get_artifacts.clear()
            st.cache_data.clear()
            st.rerun()
        st.caption("Data source: physics-inspired simulator. No credentials or phone numbers are stored in this repo.")

    tabs = st.tabs(["Overview", "Spectral health", "Trends", "Alerts", "Model"])
    with tabs[0]:
        render_overview(art)
    with tabs[1]:
        render_spectral(art)
    with tabs[2]:
        render_temporal(art)
    with tabs[3]:
        render_alerts(art)
    with tabs[4]:
        render_model(art)


if __name__ == "__main__":
    main()
