"""Check the local runtime without training data or network access."""
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def check_environment():
    cache = ROOT / ".runtime" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    errors, versions = [], {}
    if sys.version_info[:2] != (3, 12):
        errors.append("Нужен Python 3.12.")
    for filename in ("requirements.txt", "requirements-torch.txt"):
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, expected = line.strip().split("==")
            try:
                actual = metadata.version(name)
                versions[name] = actual
                if actual != expected:
                    errors.append(f"{name}: требуется {expected}, установлена {actual}.")
            except metadata.PackageNotFoundError:
                errors.append(f"Не установлен {name}.")
    if not errors:
        for module in ("torch", "numpy", "pandas", "matplotlib", "tkinter"):
            try:
                import_module(module)
            except Exception as error:
                errors.append(f"Не загружается {module}: {error}")
    return {"ready": not errors, "python": sys.version.split()[0], "versions": versions, "errors": errors}


if __name__ == "__main__":
    result = check_environment()
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(0 if result["ready"] else 1)
