"""Generation, validation and window preparation for SM7, TZ v2.

The adapter returns raw densities, never normalized values. The model's
shared input preparation layer divides X by 200 exactly once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "traffic_density.csv"
DEFAULT_GENERATOR = ROOT / "data" / "generator_config.json"
FEATURES = ("density_1", "density_2", "density_3")
COLUMNS = ("timestamp", *FEATURES)
START = datetime(2024, 1, 1)
END = datetime(2026, 1, 1)
STEP = timedelta(minutes=5)
WINDOW = 12
ROW_COUNTS = (147369, 31579, 31580)
N_ROWS = sum(ROW_COUNTS)
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\Z")


class DataError(ValueError):
    """Malformed or incompatible data."""


@dataclass(frozen=True)
class Split:
    X: np.ndarray
    y: np.ndarray
    forecast_origin: np.ndarray


@dataclass(frozen=True)
class TestSplit(Split):
    target_density: np.ndarray


def _time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is not None or value.microsecond:
            raise DataError("timestamp must have no timezone or fractional seconds")
        return value
    if not isinstance(value, str) or not TIMESTAMP_RE.fullmatch(value):
        raise DataError(f"invalid timestamp format: {value!r}; expected YYYY-MM-DDTHH:MM:SS")
    try:
        return datetime.strptime(value, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise DataError(f"invalid calendar timestamp: {value!r}") from exc


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _settings(config: dict[str, Any] | str | Path | None) -> None:
    """Reject incompatible architecture config instead of ignoring changes."""
    if config is None:
        return
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    expected = {"window_size": WINDOW, "feature_names": list(FEATURES),
                "normalization_divisor": 200, "step_minutes": 5, "horizon_minutes": 5}
    actual = config.get("input", config)
    if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
        raise DataError("architecture config is incompatible with TZ v2 (input settings)")


def _write_csv(path: Path, rows, header=COLUMNS) -> None:
    """Write atomically; do not leave a partial file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as handle:
            tmp = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        tmp.replace(path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as handle:
            tmp = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        tmp.replace(path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def generate_csv(path: str | Path = DEFAULT_CSV, generator_config: str | Path = DEFAULT_GENERATOR) -> Path:
    """Generate exactly the reproducible 210528 synthetic observations."""
    cfg = json.loads(Path(generator_config).read_text(encoding="utf-8"))
    expected = {
        "start": _iso(START), "end": _iso(END), "step_minutes": 5,
        "seeds": [43, 44, 45], "baseline_cars": 2,
        "morning_peak_hour": 8, "evening_peak_hour": 18, "peak_half_width_hours": 3,
        "weekday_factor": 1, "weekend_factor": 0.7,
        "peak_coefficients": [[10, 4], [4, 10], [7, 7]],
        "noise_cars": [-2, 2], "clip_cars": [0, 20],
        "density_multiplier": 10, "decimal_places": 1,
    }
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise DataError("generator config must match the fixed parameters of TZ v2")
    rngs = [random.Random(seed) for seed in cfg["seeds"]]

    def peak(h: float, p: int) -> float:
        return max(1 - abs(h - p) / cfg["peak_half_width_hours"], 0)

    def rows():
        for i in range(1, N_ROWS + 1):
            t = START + i * STEP
            mid = t - STEP / 2
            h = mid.hour + mid.minute / 60 + mid.second / 3600
            c = cfg["weekday_factor"] if mid.weekday() < 5 else cfg["weekend_factor"]
            morning = peak(h, cfg["morning_peak_hour"])
            evening = peak(h, cfg["evening_peak_hour"])
            density = []
            for rng, (a, b) in zip(rngs, cfg["peak_coefficients"]):
                n = cfg["baseline_cars"] + c * (a * morning + b * evening) + rng.uniform(*cfg["noise_cars"])
                n = min(max(n, cfg["clip_cars"][0]), cfg["clip_cars"][1])
                density.append(f"{cfg['density_multiplier'] * n:.{cfg['decimal_places']}f}")
            yield (_iso(t), *density)

    output = Path(path)
    _write_csv(output, rows())
    return output


def validate_csv(path: str | Path, *, full: bool = True) -> tuple[list[datetime], np.ndarray]:
    """Read strict ordered CSV; validate every row without changing the source."""
    timestamps: list[datetime] = []
    densities: list[list[float]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise DataError("duplicate column names")
        missing = set(COLUMNS) - set(columns)
        if missing:
            raise DataError(f"missing required columns: {', '.join(sorted(missing))}")
        previous = None
        for row in reader:
            line = reader.line_num
            if None in row or any(v is None or v.strip() == "" for v in row.values()):
                raise DataError(f"line {line}: missing or extra field")
            try:
                t = _time(row["timestamp"])
            except DataError as exc:
                raise DataError(f"line {line}: {exc}") from exc
            if previous is not None:
                if t == previous:
                    raise DataError(f"line {line}: duplicate timestamp")
                if t < previous:
                    raise DataError(f"line {line}: timestamps not increasing")
                if t - previous != STEP:
                    raise DataError(f"line {line}: timestamp gap; expected 5 minutes")
            previous = t
            values = []
            for col in FEATURES:
                try:
                    value = float(row[col])
                except ValueError as exc:
                    raise DataError(f"line {line}: {col} is not numeric") from exc
                if not math.isfinite(value):
                    raise DataError(f"line {line}: {col} is not finite")
                if not 0 <= value <= 200:
                    raise DataError(f"line {line}: {col} outside [0, 200]")
                if not math.isclose(value * 10, round(value * 10), rel_tol=0, abs_tol=1e-8):
                    raise DataError(f"line {line}: {col} must have at most one decimal place")
                values.append(value)
            timestamps.append(t)
            densities.append(values)
    if not timestamps:
        raise DataError("CSV contains no observations")
    if full:
        if len(timestamps) != N_ROWS:
            raise DataError(f"expected {N_ROWS} rows, found {len(timestamps)}")
        if timestamps[0] != START + STEP or timestamps[-1] != END:
            raise DataError("timestamps do not cover the exact 2024–2025 period")
    return timestamps, np.asarray(densities, dtype=np.float32)


def _windows(timestamps: list[datetime], density: np.ndarray, *, test: bool = False) -> Split:
    # Split before calling this function; targets and X cannot cross a boundary.
    n = len(density) - WINDOW
    if n < 1:
        raise DataError("a labeled split needs at least 13 rows")
    # sliding_window_view places the window axis last; transpose gives [N,12,3].
    X = np.lib.stride_tricks.sliding_window_view(density, WINDOW, axis=0)[:n]
    X = np.ascontiguousarray(X.transpose(0, 2, 1), dtype=np.float32)
    targets = density[WINDOW:]
    y = np.argmin(targets, axis=1).astype(np.int64)  # first minimum wins ties
    origins = np.asarray([_iso(t) for t in timestamps[WINDOW - 1:-1]], dtype="U19")
    if test:
        return TestSplit(X, y, origins, np.ascontiguousarray(targets, dtype=np.float32))
    return Split(X, y, origins)


def load_training_splits(path: str | Path = DEFAULT_CSV, config=None) -> tuple[Split, Split]:
    """Validate full CSV, emit only train and validation labeled windows."""
    _settings(config)
    times, density = validate_csv(path)
    a, b = ROW_COUNTS[0], sum(ROW_COUNTS[:2])
    return _windows(times[:a], density[:a]), _windows(times[a:b], density[a:b])


def load_test_split(path: str | Path = DEFAULT_CSV, config=None) -> TestSplit:
    """Only the held-out test windows, with actual next-interval densities."""
    _settings(config)
    times, density = validate_csv(path)
    b = sum(ROW_COUNTS[:2])
    return _windows(times[b:], density[b:], test=True)


def load_history(path: str | Path, t: str | datetime, config=None) -> tuple[np.ndarray, str]:
    """Return raw [12,3] history ending at requested t, and the ISO timestamp."""
    _settings(config)
    origin = _time(t)
    times, density = validate_csv(path, full=False)
    try:
        last = times.index(origin)
    except ValueError as exc:
        raise DataError(f"history has no observation ending at {_iso(origin)}") from exc
    if last + 1 < WINDOW:
        raise DataError(f"need {WINDOW} completed observations up to t; found {last + 1}")
    return density[last - WINDOW + 1:last + 1].copy(), _iso(origin)


def prepare_report(path: str | Path = DEFAULT_CSV, output: str | Path = ROOT / "results" / "data_report.json",
                   control_dir: str | Path = ROOT / "data" / "control") -> dict:
    """Produce a reproducible data QA report and the exact TZ control input."""
    times, density = validate_csv(path)
    boundaries = (0, ROW_COUNTS[0], sum(ROW_COUNTS[:2]), N_ROWS)
    splits = {}
    for name, lo, hi in zip(("train", "validation", "test"), boundaries, boundaries[1:]):
        part = _windows(times[lo:hi], density[lo:hi], test=name == "test")
        splits[name] = {"rows": hi - lo, "examples": len(part.y),
                        "first_row": _iso(times[lo]), "last_row": _iso(times[hi - 1]),
                        "first_forecast_origin": str(part.forecast_origin[0]),
                        "last_forecast_origin": str(part.forecast_origin[-1]),
                        "X_shape": list(part.X.shape), "X_dtype": str(part.X.dtype),
                        "y_shape": list(part.y.shape), "y_dtype": str(part.y.dtype)}
    b = boundaries[2]
    control = Path(control_dir)
    # Keep the target row strictly separate from the model input.
    _write_csv(control / "history.csv", ((_iso(times[i]), *(f"{v:.1f}" for v in density[i]))
                                          for i in range(b, b + WINDOW)))
    target = density[b + WINDOW]
    truth = {"forecast_origin": _iso(times[b + WINDOW - 1]),
             "target_interval_end": _iso(times[b + WINDOW]),
             "true_light": int(np.argmin(target)) + 1,
             "target_density": [round(float(v), 1) for v in target]}
    _write_json(control / "actual.json", truth)
    report = {"synthetic": True, "rows": N_ROWS,
              "first_timestamp": _iso(times[0]), "last_timestamp": _iso(times[-1]),
              "step_minutes": 5, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "density_min": round(float(density.min()), 1), "density_max": round(float(density.max()), 1),
              "splits": splits, "control": truth,
              "validation": "passed", "normalization": "not applied by adapter"}
    _write_json(Path(output), report)
    return report


def main(argv: list[str] | None = None) -> int:
    def root_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="generate synthetic two-year CSV")
    gen.add_argument("--csv", type=root_path, default=DEFAULT_CSV)
    gen.add_argument("--generator-config", type=root_path, default=DEFAULT_GENERATOR)
    val = sub.add_parser("validate", help="check CSV or working history")
    val.add_argument("--csv", type=root_path, default=DEFAULT_CSV)
    val.add_argument("--time", help="forecast origin for history validation")
    prep = sub.add_parser("prepare", help="validate and report window counts/control case")
    prep.add_argument("--csv", type=root_path, default=DEFAULT_CSV)
    prep.add_argument("--report", type=root_path, default=ROOT / "results" / "data_report.json")
    prep.add_argument("--control-dir", type=root_path, default=ROOT / "data" / "control")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            print(f"Generated: {generate_csv(args.csv, args.generator_config)}")
        elif args.command == "validate":
            if args.time:
                X, origin = load_history(args.csv, args.time)
                print(f"Valid history: {X.shape}, t={origin}")
            else:
                times, _ = validate_csv(args.csv)
                print(f"Valid CSV: {len(times)} rows, {_iso(times[0])} — {_iso(times[-1])}")
        else:
            report = prepare_report(args.csv, args.report, args.control_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError, csv.Error, ValueError, TypeError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
