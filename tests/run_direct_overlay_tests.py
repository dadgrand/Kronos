from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = Path(__file__).resolve().parent
    test_files = [
        root / "test_diplom_risk_adapter.py",
        root / "test_diplom_policy_hooks.py",
        root / "test_fresh_diplom_risk_artifact.py",
    ]
    total = 0
    for path in test_files:
        module = load_module(path)
        for name, func in sorted(vars(module).items()):
            if not name.startswith("test_") or not callable(func):
                continue
            signature = inspect.signature(func)
            if signature.parameters:
                raise RuntimeError(f"{path.name}:{name} requires unsupported direct-test parameters")
            func()
            total += 1
            print(f"ok {path.name}:{name}")
    print(f"overlay direct tests passed: {total}")


if __name__ == "__main__":
    main()
