# Membot Tests

## How to run

All tests are standalone scripts. Run them from the **membot/** directory (not from inside `tests/`):

```bash
cd membot
python tests/test_multi_cart.py
python tests/test_federate.py
```

## Convention

- **Test scripts** live in `tests/` and are tracked in git.
- **Test output** (logs, scratch directories, temporary carts) goes to subdirectories or `*.log` files. Both are gitignored — see `../.gitignore`.
- **Test data** lives under the membot data directories (`cartridges/`, etc.) — tests should not write into those.
- **Tests are non-destructive.** They use `tempfile.mkdtemp()` for any state they need, and clean up on exit.

## Path setup

Tests in this directory live one level below `membot/` so they need to add the parent to `sys.path` before importing membot modules. The pattern is:

```python
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import multi_cart  # now resolves
```

## What's here

| File | Tests |
|---|---|
| `test_multi_cart.py` | Multi-cart query layer (mount, list, search, scope, role_filter, collision, unmount) |
| `test_federate.py` | Federated mode (migrate, load_fleet, cross-fleet search, consolidate, publish, dedup) |
| `fleet/` | Pre-existing test suite from the SAGE collaboration |
