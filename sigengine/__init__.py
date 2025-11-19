"""sigengine package init

Make the sigengine module importable when running scripts from the repository root
(e.g. `python scripts/build_snapshot_feed_from_backfill.py`).

This file can be empty, but having it ensures `import sigengine.signals` works
for plain filesystem imports and avoids requiring PYTHONPATH hacks.
"""

__all__ = ["signals"]
