"""
WorldRegistry — load `worlds.toml` and resolve dotted-path class references.

A single small file: the console reads one config file, presents a dropdown,
and instantiates the chosen `WorldEnvironment` against an in-memory backend.
No magic, no auto-discovery.
"""
from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from distributed_agent_harness.world import BaseWorldEnvironment


@dataclass(frozen=True)
class WorldEntry:
    """One row of the dropdown."""
    id: str
    name: str
    project_id: str
    world_class: Type[BaseWorldEnvironment]


class WorldRegistry:
    def __init__(self, entries: list[WorldEntry]) -> None:
        self._by_id = {e.id: e for e in entries}
        if len(self._by_id) != len(entries):
            raise ValueError("Duplicate world id in registry")

    def list(self) -> list[WorldEntry]:
        return list(self._by_id.values())

    def get(self, world_id: str) -> WorldEntry:
        if world_id not in self._by_id:
            raise KeyError(f"Unknown world id: {world_id!r}")
        return self._by_id[world_id]

    @classmethod
    def from_toml(cls, path: Path) -> "WorldRegistry":
        with path.open("rb") as f:
            data = tomllib.load(f)
        entries: list[WorldEntry] = []
        for row in data.get("world", []):
            entries.append(WorldEntry(
                id=row["id"],
                name=row["name"],
                project_id=row["project_id"],
                world_class=_import_class(row["class"]),
            ))
        if not entries:
            raise ValueError(f"No [[world]] entries found in {path}")
        return cls(entries)


def _import_class(dotted: str) -> Type[BaseWorldEnvironment]:
    """Import ``module.path:ClassName`` (or ``module.path.ClassName``)."""
    if ":" in dotted:
        module_path, class_name = dotted.split(":", 1)
    else:
        module_path, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseWorldEnvironment):
        raise TypeError(f"{dotted} is not a BaseWorldEnvironment subclass")
    return cls
