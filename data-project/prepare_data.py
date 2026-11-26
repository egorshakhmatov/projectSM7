"""Prepare and validate the single 15-minute traffic-delay dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
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
    if dest.resolve() == ROOT.resolve() or (dest / 'traffic_delays.csv').exists() or (dest / 'provenance.json').exists():
        raise ValueError('Миграция требует отдельной папки без существующего датасета; канонический CSV защищён от перезаписи.')
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


def audit_external(csv_path: Path, as_of: datetime | None = None) -> dict:
    """Read-only audit of an arbitrary CSV; never accepts it into training."""
    cutoff = as_of if as_of is not None else datetime.now(timezone.utc)
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError('--as-of must include a UTC offset')
    cutoff = cutoff.astimezone(timezone.utc)
    problems: dict[str, dict] = {}

    def issue(level, code, message, line=None):
        item = problems.setdefault(code, {'level': level, 'code': code, 'message': message,
                                          'count': 0, 'example_lines': [], 'details': []})
        item['count'] += 1
        if line is not None and len(item['example_lines']) < 5:
            item['example_lines'].append(line)
        if message not in item['details'] and len(item['details']) < 10:
            item['details'].append(message)

    splits = Counter()
    observed = set()
    missing = Counter()
    values = {s: {key: [] for key in DELAYS} for s in COUNTS}
    rank = {'train': 0, 'validation': 1, 'test': 2}
    previous = None
    highest_rank = -1
    earliest = latest = None
    n = 0
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != FIELDS:
            extra = sorted(set(header) - set(FIELDS))
            absent = sorted(set(FIELDS) - set(header))
            issue('error', 'SCHEMA', f'Нужны ровно семь полей в порядке {FIELDS}; лишние: {extra}; отсутствуют: {absent}. Лишние признаки могут содержать будущую информацию.')
            if absent or len(set(header)) != len(header):
                return {'status': 'rejected', 'rows': 0, 'as_of_utc': cutoff.isoformat(),
                        'issues': list(problems.values()), 'metrics': {}}
        for line, row in enumerate(reader, 2):
            n += 1
            if None in row or any(row.get(key) is None or not row[key].strip() for key in FIELDS):
                issue('error', 'MISSING_FIELDS', 'Пустые значения или неверное количество полей.', line)
                for key in FIELDS:
                    if row.get(key) is None or not row[key].strip():
                        missing[key] += 1
            split = row.get('split')
            if split not in rank:
                issue('error', 'SPLIT_VALUE', 'Выборка должна быть train, validation или test.', line)
            else:
                splits[split] += 1
                if rank[split] < highest_rank:
                    issue('error', 'SPLIT_LEAKAGE', 'Нарушен порядок train → validation → test: будущая выборка расположена раньше прошлой.', line)
                highest_rank = max(highest_rank, rank[split])
            try:
                local = datetime.fromisoformat((row.get('timestamp') or '').replace('Z', '+00:00'))
                if local.tzinfo is None or local.utcoffset() is None:
                    raise ValueError('missing UTC offset')
                stamp = local.astimezone(timezone.utc)
            except (ValueError, OverflowError):
                issue('error', 'TIMESTAMP', 'Дата/время должны быть в ISO 8601 с часовым поясом.', line)
                local = stamp = None
            if stamp is not None:
                earliest = stamp if earliest is None else min(earliest, stamp)
                latest = stamp if latest is None else max(latest, stamp)
                if stamp > cutoff:
                    issue('error', 'FUTURE_OBSERVATION', 'Наблюдение позже даты отсечения: будущие фактические задержки недопустимы.', line)
                if stamp in observed:
                    issue('error', 'DUPLICATE_TIMESTAMP', 'Повторяющаяся временная отметка в UTC.', line)
                observed.add(stamp)
                if stamp.minute % 15 or stamp.second or stamp.microsecond:
                    issue('error', 'OFF_GRID', 'Отметка не лежит на 15-минутной сетке UTC.', line)
                if previous is not None:
                    if stamp < previous:
                        issue('error', 'TIME_ORDER', 'Нарушена хронология; строки нельзя перемешивать.', line)
                    elif stamp - previous != STEP:
                        issue('error', 'TIME_STEP', 'Пропуск либо неверный временной шаг (ожидается 15 минут).', line)
                previous = stamp
                if row.get('is_weekend') in ('0', '1') and row['is_weekend'] != str(int(local.weekday() >= 5)):
                    issue('error', 'WEEKEND_MISMATCH', 'Признак выходного не согласуется с локальной датой.', line)
            for key in ('is_holiday', 'is_weekend'):
                if row.get(key) not in ('0', '1'):
                    issue('error', 'CALENDAR_FLAG', f'{key}: допустимы только 0 и 1.', line)
            for key in DELAYS:
                try:
                    value = float(row.get(key, ''))
                    if not math.isfinite(value) or value < 0:
                        raise ValueError('invalid delay')
                except (ValueError, TypeError):
                    issue('error', 'INVALID_DELAY', f'{key}: задержки должны быть конечными, числовыми и неотрицательными.', line)
                    continue
                if split in values:
                    values[split][key].append(value)
    if not n:
        issue('error', 'EMPTY', 'Нет наблюдений.')
    for split in COUNTS:
        if splits[split] < 12:
            issue('error', 'INSUFFICIENT_WINDOW', f'{split}: необходимо не менее 12 последовательных строк для окна 8 + горизонт 4.')
        if n and abs(splits[split] / n - {'train': .70, 'validation': .15, 'test': .15}[split]) > .05:
            issue('warning', 'SPLIT_RATIO', f'{split}: доля отклоняется более чем на 5 процентных пунктов от 70/15/15.')

    metrics = {}
    for split, columns in values.items():
        metrics[split] = {'rows': splits[split], 'share': round(splits[split] / n, 4) if n else 0,
                          'delay_minutes': {}}
        for key, data in columns.items():
            if not data:
                continue
            ordered = sorted(data)
            p95 = ordered[math.ceil(.95 * len(ordered)) - 1]
            middle = len(ordered) // 2
            median = ordered[middle] if len(ordered) % 2 else ordered[middle - 1] / 2 + ordered[middle] / 2
            metrics[split]['delay_minutes'][key] = {
                'n': len(data), 'min': ordered[0], 'mean': round(statistics.mean(data), 5),
                'median': round(median, 5), 'p95': p95, 'max': ordered[-1],
                'zero_share': round(sum(v == 0 for v in data) / len(data), 5),
            }
            if ordered[0] == ordered[-1]:
                issue('warning', 'CONSTANT_DELAY', f'{split}/{key}: все задержки одинаковые; проверьте источник.')
            if ordered[-1] > 60:
                issue('warning', 'LARGE_DELAY', f'{split}/{key}: максимум больше 60 минут. Это ориентир для ручной проверки, а не физический предел.')
    for split in ('validation', 'test'):
        for key in DELAYS:
            train = metrics['train']['delay_minutes'].get(key)
            other = metrics[split]['delay_minutes'].get(key)
            if train and other and abs(other['median'] - train['median']) > max(0.1, train['p95']):
                issue('warning', 'DISTRIBUTION_SHIFT', f'{split}/{key}: сильное изменение медианы относительно train (эвристика, не доказательство ошибки).')

    issue('warning', 'PROVENANCE_REQUIRED', 'Для стороннего CSV происхождение строк и способ расчёта задержек не подтверждены. Нельзя гарантировать отсутствие утечки в самих значениях или окон на границе источников. Аудит не подключает файл к обучению.')
    issue('warning', 'HOLIDAY_CALENDAR', 'Праздники проверены только на формат 0/1: подтвердите регион и календарь источника.')
    rejected = any(item['level'] == 'error' for item in problems.values())
    if not rejected:
        for split in COUNTS:
            metrics[split]['candidate_windows_before_provenance'] = max(0, splits[split] - 11)
    return {'status': 'rejected' if rejected else 'manual_review_required', 'rows': n,
            'columns': list(header), 'as_of_utc': cutoff.isoformat(),
            'first_utc': earliest.isoformat() if earliest else None,
            'last_utc': latest.isoformat() if latest else None,
            'unique_timestamps': len(observed), 'missing_by_column': dict(missing),
            'issues': list(problems.values()), 'metrics': metrics}


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Previous full CSV (optional; migration only)')
    parser.add_argument('--directory', type=Path, default=ROOT)
    parser.add_argument('--audit', type=Path, help='Audit another CSV without changing the project dataset')
    parser.add_argument('--as-of', type=datetime.fromisoformat, help='Latest allowed historical timestamp (ISO with UTC offset)')
    args = parser.parse_args()
    try:
        if args.audit:
            if args.source:
                parser.error('--audit and --source cannot be combined')
            report = audit_external(args.audit, args.as_of)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise SystemExit(1 if report['status'] == 'rejected' else 0)
        if args.as_of is not None:
            parser.error('--as-of requires --audit')
        if args.source:
            compact(args.source, args.directory)
        validate(args.directory)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f'ОШИБКА: {error}', file=sys.stderr)
        raise SystemExit(1)
