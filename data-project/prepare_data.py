"""Prepare and validate the single 15-minute traffic-delay dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ('timestamp', 'is_holiday', 'is_weekend', 'delay_A', 'delay_B', 'delay_C', 'split')
DELAYS = ('delay_A', 'delay_B', 'delay_C')
COUNTS = {'train': 49123, 'validation': 10526, 'test': 10527}
WINDOW_COUNTS = {'train': 45040, 'validation': 9746, 'test': 9852}
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def csv_digest(path: Path) -> str:
    # Git may convert LF to CRLF on Windows. Hash the canonical LF bytes.
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def compact(source: Path, dest: Path) -> None:
    """Convert the previous enriched CSV to seven columns without changing observations."""
    with source.open(encoding='utf-8-sig', newline='') as f:
        original = list(csv.DictReader(f))
    source_labels = [row['source'] for row in original]
    expected_flags = [int(row['window_ok']) for row in original]
    compact_rows = []
    for row in original:
        offset = int(row['utc_offset_minutes'])
        stamp = datetime.fromisoformat(row['timestamp_utc'].replace('Z', '+00:00'))
        local = stamp.astimezone(timezone(timedelta(minutes=offset)))
        if local.date().isoformat() != row['date'] or local.strftime('%H:%M') != row['time']:
            raise ValueError('Previous calendar fields do not match the timestamp')
        compact_rows.append({'timestamp': local.isoformat(timespec='seconds'),
                             **{field: row[field] for field in FIELDS[1:]}})
    check_rows(compact_rows)
    if list(valid_origins(compact_rows, source_labels)) != [i for i, flag in enumerate(expected_flags) if flag]:
        raise ValueError('Window boundaries changed during conversion')
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / 'traffic_delays.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(compact_rows)
    start = 0
    observed_ranges = []
    for label, group in groupby(source_labels):
        end = start + sum(1 for _ in group)
        if label == 'observed_proxy':
            observed_ranges.append([start, end])
        elif label != 'synthetic':
            raise ValueError(f'Unknown source: {label}')
        start = end
    metadata = {
        'schema_version': 1,
        'previous_full_dataset_commit': 'aa4144b5c6389add293939e02d0e31b5d51d0c60',
        'csv_sha256_lf': csv_digest(csv_path),
        'rows': len(compact_rows),
        'range_convention': 'zero-based, end exclusive; other rows are synthetic',
        'observed_proxy_ranges': observed_ranges,
        'source_counts': dict(Counter(source_labels)),
        'window_counts': dict(Counter(compact_rows[i]['split'] for i in valid_origins(compact_rows, source_labels))),
    }
    (dest / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def check_rows(rows: list[dict]) -> None:
    if len(rows) != sum(COUNTS.values()):
        raise ValueError(f'Unexpected row count: {len(rows)}')
    for i, row in enumerate(rows):
        stamp = datetime.fromisoformat(row['timestamp'])
        if stamp.utcoffset() not in (timedelta(hours=-5), timedelta(hours=-6)):
            raise ValueError(f'Incorrect Chicago UTC offset at row {i+2}')
        if stamp.astimezone(timezone.utc) != START + STEP * i:
            raise ValueError(f'Missing or duplicate 15-minute interval at row {i+2}')
        if row['is_holiday'] not in ('0', '1') or row['is_weekend'] != str(int(stamp.weekday() >= 5)):
            raise ValueError(f'Invalid calendar flag at row {i+2}')
        if any(not math.isfinite(float(row[key])) or float(row[key]) < 0 for key in DELAYS):
            raise ValueError(f'Invalid delay at row {i+2}')
        expected = 'train' if i < COUNTS['train'] else 'validation' if i < COUNTS['train'] + COUNTS['validation'] else 'test'
        if row['split'] != expected:
            raise ValueError(f'Unexpected split at row {i+2}')


def valid_origins(rows: list[dict], labels: list[str]):
    """Only windows with 8 history + 4 future steps in one source and split."""
    for i in range(7, len(rows) - 4):
        if all(rows[j]['split'] == rows[i]['split'] and labels[j] == labels[i] for j in range(i - 7, i + 5)):
            yield i


def load_data(directory: Path = ROOT):
    csv_path = directory / 'traffic_delays.csv'
    with csv_path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError('Expected exactly the seven documented columns')
        rows = list(reader)
    check_rows(rows)
    metadata = json.loads((directory / 'provenance.json').read_text(encoding='utf-8'))
    if metadata['csv_sha256_lf'] != csv_digest(csv_path) or metadata['rows'] != len(rows):
        raise ValueError('Provenance is not associated with this CSV version')
    labels = ['synthetic'] * len(rows)
    last_end = 0
    for start, end in metadata['observed_proxy_ranges']:
        if not last_end <= start < end <= len(rows):
            raise ValueError('Invalid provenance range')
        labels[start:end] = ['observed_proxy'] * (end - start)
        last_end = end
    counts = dict(Counter(labels))
    if counts != {'synthetic': 49123, 'observed_proxy': 21053} or counts != metadata['source_counts']:
        raise ValueError('Unexpected source counts')
    return rows, labels


def load_split(split: str, directory: Path = ROOT):
    """Return X [N,8,3], y [N,3], and source labels as Python lists."""
    if split not in COUNTS:
        raise ValueError(f'Unknown split: {split}')
    rows, labels = load_data(directory)
    delays = [[float(row[key]) for key in DELAYS] for row in rows]
    origins = [i for i in valid_origins(rows, labels) if rows[i]['split'] == split]
    return [delays[i-7:i+1] for i in origins], [delays[i+4] for i in origins], [labels[i] for i in origins]


def validate(directory: Path = ROOT):
    rows, labels = load_data(directory)
    counts = dict(Counter(rows[i]['split'] for i in valid_origins(rows, labels)))
    if counts != WINDOW_COUNTS:
        raise ValueError(f'Unexpected valid window counts: {counts}')
    print(json.dumps({'rows': len(rows), 'columns': list(FIELDS), 'splits': dict(Counter(r['split'] for r in rows)),
                      'sources': dict(Counter(labels)), 'valid_windows': counts}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Previous full CSV (optional; migration only)')
    parser.add_argument('--directory', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.source:
        compact(args.source, args.directory)
    validate(args.directory)
