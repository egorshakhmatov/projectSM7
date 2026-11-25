"""Entry point for the architecture phase. Does not generate or modify data."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import webbrowser


ROOT = Path(__file__).resolve().parent
PLANNED = {
    "generate": "Генератор подключит ответственный за данные.",
    "prepare": "Подготовку выборок подключит ответственный за данные.",
    "train": "Обучение подключим после согласования модулей и получения данных.",
    "evaluate": "Оценка станет доступна после подключения модели и тестовой выборки.",
    "predict": "Для прогноза нужны модуль прогнозирования и обученные веса.",
}


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def get_status() -> dict:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    return {
        "phase": "architecture",
        "python_version": sys.version.split()[0],
        "python_matches_spec": sys.version_info[:2] == (3, 12),
        "model_design": config["model"],
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "implemented_commands": ["architecture", "model", "status"],
        "pending_commands": list(PLANNED),
        "forecast_available": False,
        "artifacts": {
            key: {"path": value, "exists": (ROOT / value).is_file()}
            for key, value in config["paths"].items()
            if key != "results"
        },
        "note": "Наличие файла не означает, что проверены его формат и содержимое.",
    }


def show_status(as_json: bool = False) -> int:
    status = get_status()
    if as_json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    print("СМ7 — этап проектирования архитектуры")
    print(f"Python: {status['python_version']}")
    print("Версия по ТЗ: " + ("совпадает" if status["python_matches_spec"] else "нужна 3.12"))
    print("Выбранная модель: GRU, один слой, 32 скрытых признака, 3 класса.")
    print("Работают: меню, просмотр архитектуры, объяснение модели, этот отчёт.")
    print("Обучение и прогноз пока не подключены.")
    print("PyTorch: " + ("найден" if status["torch_installed"] else "не установлен; для меню не нужен"))
    for item in status["artifacts"].values():
        label = "файл найден" if item["exists"] else "файл пока отсутствует"
        print(f"  {item['path']}: {label}")
    print(status["note"])
    return 0


def show_architecture() -> int:
    document = ROOT / "docs" / "overview.html"
    if webbrowser.open(document.as_uri()):
        print("Схема архитектуры открыта в браузере.")
        return 0
    print(f"Откройте схему вручную: {document}", file=sys.stderr)
    return 1


def show_model() -> int:
    print((ROOT / "docs" / "model-choice.md").read_text(encoding="utf-8"))
    return 0


def menu() -> int:
    if not sys.stdin.isatty():
        return show_status()
    while True:
        print("\nСМ7 — архитектура проекта")
        print("1. Открыть схему архитектуры")
        print("2. Прочитать, почему выбрана GRU")
        print("3. Посмотреть состояние проекта")
        print("0. Выход")
        try:
            choice = input("Выберите пункт: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "0":
            return 0
        action = {"1": show_architecture, "2": show_model, "3": show_status}.get(choice)
        if action:
            action()
        else:
            print("Введите 0, 1, 2 или 3.")


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="СМ7: единая точка входа в каркас проекта.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("architecture", help="открыть схему в браузере")
    commands.add_parser("model", help="показать обоснование модели")
    status = commands.add_parser("status", help="показать состояние каркаса")
    status.add_argument("--json", action="store_true", help="вывести машиночитаемый отчёт")
    for name, explanation in PLANNED.items():
        commands.add_parser(name, help=f"ещё не подключено: {explanation}")
    args = parser.parse_args()
    if args.command in PLANNED:
        print(f"Режим {args.command} ещё не подключён. {PLANNED[args.command]}", file=sys.stderr)
        return 2
    actions = {"architecture": show_architecture, "model": show_model}
    if args.command == "status":
        return show_status(args.json)
    if args.command in actions:
        return actions[args.command]()
    return menu()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Не удалось запустить каркас: {error}", file=sys.stderr)
        raise SystemExit(1)
