"""Name -> detector registry, so the bounding box model is a *value*.

Nothing downstream should mention a detector class by name: `ShipCropper`,
main.py and (from step 09) the training config all say which model they want as
a string or a dict, and this module hands back the object.

    build_detector("yolov3", conf_thresh=0.3)
    build_detector({"name": "arete", "box": "loose"})
    build_detector("mypkg.detectors:Yolov8ShipDetector", weights="best.pt")

The last form is the escape hatch: any class satisfying the `ShipDetector`
contract can be dropped in from outside this repo without editing realsimir at
all.  Entries are registered lazily ("module:Class" strings, imported on first
use), so listing the available models never imports torch.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from .base import ShipDetector

__all__ = ["register_detector", "register_lazy", "build_detector", "available_detectors", "get_detector_class"]

# name -> class, or "module:Class" while still unimported
_REGISTRY: dict[str, type | str] = {}
_ALIASES: dict[str, str] = {}
_DOC: dict[str, str] = {}


def _norm(name: str) -> str:
    return name.strip().lower()


def _resolve_name(name: str) -> str:
    key = _norm(name)
    key = _ALIASES.get(key, key)
    if key not in _REGISTRY:
        known = ", ".join(available_detectors())
        raise KeyError(f"unknown detector {name!r}; registered: {known} (or pass 'module:ClassName')")
    return key


def register_detector(name: str, *aliases: str, doc: str | None = None):
    """Class decorator: make `name` build this detector.

        @register_detector("yolov8", "yolo8")
        class Yolov8ShipDetector(ShipDetector): ...
    """

    def deco(cls: type) -> type:
        _put(name, cls, aliases, doc or (cls.__doc__ or "").strip().split("\n")[0])
        return cls

    return deco


def register_lazy(name: str, target: str, *aliases: str, doc: str = "") -> None:
    """Register "module:Class" without importing it.

    Used for the built-ins so that `import realsimir` stays cheap and a broken
    optional dependency only bites the model that needs it.
    """
    _put(name, target, aliases, doc)


def _put(name: str, target: type | str, aliases: tuple[str, ...], doc: str) -> None:
    key = _norm(name)
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not target and not _same_target(existing, target):
        raise ValueError(f"detector {name!r} is already registered to {existing}")
    # a class beats the lazy "module:Class" string that named it
    if not (isinstance(target, str) and isinstance(existing, type)):
        _REGISTRY[key] = target
    if doc and not _DOC.get(key):
        _DOC[key] = doc
    for a in aliases:
        _ALIASES[_norm(a)] = key


def _same_target(a: type | str, b: type | str) -> bool:
    """True when a class and a 'module:Class' string name the same thing.

    Built-ins are registered twice on purpose: lazily in __init__.py (so the
    registry lists them without importing torch) and by the @register_detector
    decorator on the class itself (so the module reads as self-contained, and
    so out-of-tree modules can use the same idiom).
    """

    def as_str(x):
        return f"{x.__module__}:{x.__qualname__}" if isinstance(x, type) else x

    return as_str(a) == as_str(b)


def available_detectors() -> list[str]:
    return sorted(_REGISTRY)


def describe_detectors() -> dict[str, str]:
    """name -> one-line description, for --help text and `python -m realsimir.detectors list`."""
    return {k: _DOC.get(k, "") for k in available_detectors()}


def _import_target(target: str) -> type:
    mod_name, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:ClassName', got {target!r}")
    cls = getattr(importlib.import_module(mod_name), attr)
    if not (isinstance(cls, type) and issubclass(cls, ShipDetector)):
        raise TypeError(f"{target} is not a ShipDetector subclass")
    return cls


def get_detector_class(name: str) -> type:
    """The class behind a registered name or a 'module:ClassName' path."""
    if ":" in name or "/" in name:
        return _import_target(name)
    key = _resolve_name(name)
    target = _REGISTRY[key]
    if isinstance(target, str):
        target = _import_target(target)
        _REGISTRY[key] = target  # memoise the import
    return target


def build_detector(spec: Any = "yolov3", **overrides) -> ShipDetector:
    """Turn a name / config dict / class / instance into a live detector.

    `overrides` win over anything in a config dict, which is what lets the CLI
    layer on `--conf 0.3` over whatever the config file said.
    """
    if isinstance(spec, ShipDetector):
        if overrides:
            raise TypeError("build_detector got keyword overrides for an already-built detector")
        return spec

    if isinstance(spec, Mapping):
        cfg = dict(spec)
        name = cfg.pop("name", None) or cfg.pop("type", None)
        if name is None:
            raise ValueError("detector config needs a 'name' key")
        return build_detector(name, **{**cfg, **overrides})

    cls = spec if isinstance(spec, type) else get_detector_class(str(spec))
    if not issubclass(cls, ShipDetector):
        raise TypeError(f"{cls.__name__} is not a ShipDetector subclass")
    return cls(**overrides)
