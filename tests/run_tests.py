# Zero-dependency test runner: discovers test_*.py and reports pass/fail/skip.
import importlib.util
import inspect
import os
import sys
import traceback
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def _load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    files = sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    npass = nfail = nskip = 0
    failures = []
    for f in files:
        try:
            mod = _load(os.path.join(HERE, f))
        except Exception as ex:
            print(f"  [ERROR loading] {f}: {ex}"); nfail += 1; failures.append((f, ex)); continue
        tests = [(n, fn) for n, fn in inspect.getmembers(mod, inspect.isfunction)
                 if n.startswith("test_") and fn.__module__ == mod.__name__]
        for name, fn in tests:
            label = f"{f}::{name}"
            try:
                fn()
                print(f"  PASS  {label}"); npass += 1
            except unittest.SkipTest as sk:
                print(f"  SKIP  {label}  ({sk})"); nskip += 1
            except Exception as ex:
                print(f"  FAIL  {label}  {type(ex).__name__}: {ex}")
                failures.append((label, ex)); nfail += 1
    print(f"\n{npass} passed, {nfail} failed, {nskip} skipped")
    if failures:
        print("\nfirst failure detail:")
        traceback.print_exception(type(failures[0][1]), failures[0][1], failures[0][1].__traceback__)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
