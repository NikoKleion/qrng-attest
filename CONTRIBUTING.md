# Contributing

Issues and pull requests are welcome.

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest tests
python tests/run_tests.py    # the same suite with no pytest dependency
```

The NIST known-answer tests in `tests/test_nist_kat.py` are the ones that matter most: they pin the
estimators to NIST's published vectors to under 1e-9 bits. Any change that moves those numbers is a
regression unless it is moving them toward the reference.

## Conventions

- Every entropy bound is one-sided and must never over-credit. Tests are written as
  `attested <= truth`, not `attested ~= truth`. A change that raises a reported bound needs a proof or a
  measurement behind it.
- New estimators or tests that reimplement a standard are ported from the reference implementation and
  validated against published vectors, not written from prose.
- Comments mark headers and functions. Explanation of why something is built a particular way belongs in
  the pull request description or the docs, not inline.
