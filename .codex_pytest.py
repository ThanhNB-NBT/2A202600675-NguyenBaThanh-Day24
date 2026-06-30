from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
import traceback


def _load_module(path: pathlib.Path):
    name = "test_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _test_files(args: list[str]) -> list[pathlib.Path]:
    paths = [pathlib.Path(a) for a in args if not a.startswith("-") and a != "tests/"]
    if not paths:
        paths = [pathlib.Path("tests")]
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("test_*.py")))
        elif path.name.startswith("test_") and path.suffix == ".py":
            files.append(path)
    return files


def main() -> int:
    total = failed = 0
    for file in _test_files(sys.argv[1:]):
        module = _load_module(file)
        fixtures = {
            name: value
            for name, value in vars(module).items()
            if callable(value) and getattr(value, "__pytest_fixture__", False)
        }
        for name, test in vars(module).items():
            if not name.startswith("test_") or not callable(test):
                continue
            total += 1
            try:
                kwargs = {
                    param: fixtures[param]()
                    for param in inspect.signature(test).parameters
                }
                test(**kwargs)
            except Exception:
                failed += 1
                print(f"FAILED {file}:{name}")
                traceback.print_exc()
    passed = total - failed
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
