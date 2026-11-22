"""Command line entry points.

    cropmon pipeline     # full analysis, writes artifacts and figures
    cropmon train        # retrain and print holdout metrics
    cropmon demo-data    # write raw simulated CSVs without training
    cropmon alerts       # list alerts, optionally deliver them
    cropmon serve        # run the FastAPI service
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from cropmon.alerts.channels import build_channel
from cropmon.alerts.rules import Alert, AlertEngine
from cropmon.config import Settings
from cropmon.data.synthetic import simulate_field, simulate_history, simulate_sensors
from cropmon.features.indices import compute_indices
from cropmon.pipeline import load_artifacts, run_pipeline


def _cmd_pipeline(args: argparse.Namespace) -> int:
    result = run_pipeline(Settings.from_env(), make_figures=not args.no_figures)
    print(json.dumps(result.summary, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    result = run_pipeline(Settings.from_env(), make_figures=False)
    metrics = result.metrics
    print(f"accuracy  {metrics['accuracy']:.4f}")
    print(f"macro F1  {metrics['macro_f1']:.4f}")
    print(f"ROC AUC   {metrics['roc_auc_ovr']:.4f}")
    print(f"model     {result.settings.model_path}")
    print("top features:", ", ".join(row["feature"] for row in metrics["feature_importance"][:5]))
    return 0


def _cmd_demo_data(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    settings.ensure_dirs()
    field = simulate_field(settings.grid_rows, settings.grid_cols, settings.seed)
    sensors = simulate_sensors(settings.sensor_days, settings.seed)
    indices = compute_indices(field.cube, field.wavelengths)

    np.savez_compressed(settings.data_dir / "demo_spectral_cube.npz", cube=field.cube, wavelengths=field.wavelengths)
    sensors.to_csv(settings.data_dir / "sensors.csv", index=False)

    from cropmon.analysis.zones import summarise_map, zone_map

    zones = zone_map(field.shape, settings.zone_rows, settings.zone_cols)
    base = summarise_map(indices["ndvi"], zones)["mean"].to_numpy()
    history = simulate_history(base, settings.history_weeks, settings.seed)
    history.to_csv(settings.data_dir / "zone_history.csv", index=False)

    print(f"field       {field.shape[0]}x{field.shape[1]}x{field.cube.shape[-1]} bands")
    print(f"sensors     {len(sensors)} hourly rows -> {settings.data_dir / 'sensors.csv'}")
    print(f"history     {len(history)} zone-week rows -> {settings.data_dir / 'zone_history.csv'}")
    print(f"cube        {settings.data_dir / 'demo_spectral_cube.npz'}")
    return 0


def _cmd_alerts(args: argparse.Namespace) -> int:
    art = load_artifacts(Settings.from_env())
    alerts = [Alert(**{k: v for k, v in item.items() if k != "alert_id"}) for item in art["alerts"]]
    alerts = [alert for alert in alerts if args.severity is None or alert.severity == args.severity]
    alerts = sorted(alerts, key=lambda alert: alert.severity, reverse=True)[: args.limit]

    for alert in alerts:
        print(alert.sms_text())
    print(f"\n{len(alerts)} alert(s) listed")

    if not args.send:
        print("dry run: pass --send to deliver through the configured channel")
        return 0
    channel = build_channel(args.channel)
    engine = AlertEngine(cooldown_hours=0)
    delivered = engine.dispatch(
        alerts,
        sender=lambda alert: channel.send(alert.sms_text(), to=args.to).ok,
    )
    print(f"delivered {len(delivered)} via {channel.name}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("cropmon.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cropmon", description="AI crop monitoring platform")
    sub = parser.add_subparsers(dest="command", required=True)

    pipeline = sub.add_parser("pipeline", help="run the full analysis pipeline")
    pipeline.add_argument("--no-figures", action="store_true", help="skip PNG rendering")
    pipeline.set_defaults(func=_cmd_pipeline)

    train = sub.add_parser("train", help="retrain the fused risk model")
    train.set_defaults(func=_cmd_train)

    demo = sub.add_parser("demo-data", help="write raw simulated datasets")
    demo.set_defaults(func=_cmd_demo_data)

    alerts = sub.add_parser("alerts", help="list or deliver alerts")
    alerts.add_argument("--severity", choices=["info", "warning", "critical"])
    alerts.add_argument("--limit", type=int, default=20)
    alerts.add_argument("--channel", default=None, help="console (default), twilio, fast2sms")
    alerts.add_argument("--to", default=None, help="E.164 recipient")
    alerts.add_argument("--send", action="store_true", help="actually deliver messages")
    alerts.set_defaults(func=_cmd_alerts)

    serve = sub.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
