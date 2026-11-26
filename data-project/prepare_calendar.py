"""Enrich the canonical 15-minute delay dataset without changing observations."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=15)
SPLIT_COUNTS = {"train": 49123, "validation": 10526, "test": 10527}
CALENDAR_FIELDS = (
    "date", "day", "month", "year", "day_of_week", "time",
    "is_weekend", "is_holiday", "is_nonworking_day", "utc_offset_minutes",
)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def observed(day: date) -> date:
    """Standard Monday-Friday federal holiday observance."""
    return day + timedelta(days={5: -1, 6: 1}.get(day.weekday(), 0))


def holidays(year: int) -> set[date]:
    """Recurring nationwide US federal holidays, excluding one-off closures."""
    fixed = [(1, 1), (6, 19), (7, 4), (11, 11), (12, 25)]
    fixed_observed = {observed(date(y, m, d)) for y in (year, year + 1) for m, d in fixed}
    moving = {
        nth_weekday(year, 1, 0, 3),  # MLK Day
        nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        date(year, 5, 31) - timedelta(days=date(year, 5, 31).weekday()),
        nth_weekday(year, 9, 0, 1),  # Labor Day
        nth_weekday(year, 10, 0, 2),  # Columbus Day
        nth_weekday(year, 11, 3, 4),  # Thanksgiving
    }
    return moving | {d for d in fixed_observed if d.year == year}


def chicago_local(instant: datetime) -> datetime:
    """Chicago CST/CDT rules for 2024–2025; UTC remains the unique time key."""
    if instant.tzinfo != UTC or not START <= instant < END:
        raise ValueError("Timestamp outside the supported 2024–2025 UTC period")
    march = nth_weekday(instant.year, 3, 6, 2)
    november = nth_weekday(instant.year, 11, 6, 1)
    dst_start = datetime(march.year, march.month, march.day, 8, tzinfo=UTC)
    dst_end = datetime(november.year, november.month, november.day, 7, tzinfo=UTC)
    offset = -5 if dst_start <= instant < dst_end else -6
    return instant.astimezone(timezone(timedelta(hours=offset)))


def calendar_features(instant: datetime) -> dict[str, str]:
    local = chicago_local(instant)
    day = local.date()
    weekend = int(day.weekday() >= 5)
    holiday = int(day in holidays(day.year))
    return {
        "date": day.isoformat(), "day": str(day.day), "month": str(day.month),
        "year": str(day.year), "day_of_week": str(day.isoweekday()),
        "time": local.strftime("%H:%M"), "is_weekend": str(weekend),
        "is_holiday": str(holiday), "is_nonworking_day": str(int(weekend or holiday)),
        "utc_offset_minutes": str(int(local.utcoffset().total_seconds() // 60)),
    }


def enrich(source: Path, target: Path) -> dict[str, int]:
    """Preserve source values, enforce chronological splits, append split last."""
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    expected_time = START
    temp_path: Path | None = None
    try:
        with source.open(encoding="utf-8-sig", newline="") as src:
            reader = csv.DictReader(src)
            if not reader.fieldnames or not {"timestamp_utc", "delay_A", "delay_B", "delay_C", "split"} <= set(reader.fieldnames):
                raise ValueError("Missing required delay dataset columns")
            original = [x for x in reader.fieldnames if x not in CALENDAR_FIELDS and x not in {"timestamp_utc", "split"}]
            fields = ["timestamp_utc", *CALENDAR_FIELDS, *original, "split"]
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=target.parent, delete=False) as tmp:
                temp_path = Path(tmp.name)
                writer = csv.DictWriter(tmp, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for number, row in enumerate(reader):
                    instant = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
                    if instant != expected_time:
                        raise ValueError(f"Timestamp gap/duplicate at row {number + 2}: {instant}")
                    expected_time += STEP
                    expected_split = "train" if number < 49123 else "validation" if number < 59649 else "test"
                    if row["split"] != expected_split:
                        raise ValueError(f"Unexpected split at row {number + 2}")
                    counts[row["split"]] += 1
                    row.update(calendar_features(instant))
                    writer.writerow(row)
        if expected_time != END or dict(counts) != SPLIT_COUNTS:
            raise ValueError(f"Incomplete dataset: {dict(counts)}")
        temp_path.replace(target)
        return dict(counts)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original traffic_delays.csv")
    parser.add_argument("target", type=Path, help="Output CSV; may be the same file")
    args = parser.parse_args()
    print(enrich(args.source, args.target))
