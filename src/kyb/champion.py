"""The champion reader: the model every screen uses unless told otherwise.

It changes only by promotion from a reader eval, and every promotion is kept in
config/champion.json (who replaced whom, from which run, with which weights and scores), so a
reviewer can see why the production reader is what it is. Until the first promotion, the
champion is `[defaults] reader` in config/models.toml.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

CHAMPION = Path("config/champion.json")


def current(cfg: dict, path: Path = CHAMPION) -> str:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["reader"]
    return cfg.get("defaults", {}).get("reader", "")


def history(path: Path = CHAMPION) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("history", []) if path.exists() else []


def promote(reader: str, *, replaced: str, run: str, weights: dict, scores: dict,
            path: Path = CHAMPION) -> dict:
    """Make `reader` the champion and record why."""
    entry = {"reader": reader, "replaced": replaced, "promoted_at": datetime.now().isoformat(timespec="seconds"),
             "run": run, "weights": weights, "scores": scores}
    data = {"reader": reader, "history": [*history(path), entry]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return entry
