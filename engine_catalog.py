from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# One registry owns the UI-facing metadata for every engine. Runtime availability
# still comes from engine_paths.engine_status(), so adding/removing an installed
# engine never requires editing a browser-side option list.
ENGINE_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "rife",
        "role": "interpolator",
        "label": "RIFE",
        "hint": "Fast conventional temporal interpolation for smooth in-betweens.",
        "generative": False,
        "order": 10,
    },
    {
        "id": "amt",
        "role": "interpolator",
        "label": "AMT",
        "hint": "Conventional temporal interpolation with strong low-VRAM fallbacks.",
        "generative": False,
        "order": 20,
    },
    {
        "id": "resshift",
        "role": "interpolator",
        "label": "Multi-Input ResShift Diffusion",
        "hint": "Endpoint-constrained residual diffusion for difficult occlusion, articulation and missing-content transitions.",
        "generative": True,
        "order": 30,
    },
    {
        "id": "mog_ani",
        "role": "interpolator",
        "label": "MoG Animation",
        "hint": "Motion-aware generative interpolation tuned for animation.",
        "generative": True,
        "order": 40,
        "low_vram_warning_gb": 6,
    },
    {
        "id": "mog_real",
        "role": "interpolator",
        "label": "MoG Real-world",
        "hint": "Motion-aware generative interpolation tuned for photographic and real-world footage.",
        "generative": True,
        "order": 50,
        "low_vram_warning_gb": 6,
    },
    {
        "id": "tooncrafter",
        "role": "interpolator",
        "label": "ToonCrafter (cartoon/anime)",
        "hint": "Generative cartoon interpolation that creates a transition from the two endpoint frames and selects the requested in-betweens.",
        "generative": True,
        "order": 60,
        "low_vram_warning_gb": 6,
    },
    {
        "id": "eden",
        "role": "generator",
        "label": "EDEN",
        "hint": "Structural midpoint frame generation. It can be combined with an interpolator for further filling.",
        "generative": True,
        "order": 10,
    },
    {
        "id": "speed",
        "role": "generator",
        "label": "SPEED",
        "hint": "Structural midpoint frame generation. It can be combined with an interpolator for further filling.",
        "generative": True,
        "order": 20,
    },
)

_ENGINE_BY_ID = {
    str(definition["id"]).strip().lower(): definition
    for definition in ENGINE_REGISTRY
}


def engine_definition(engine_id: object) -> dict[str, Any] | None:
    definition = _ENGINE_BY_ID.get(str(engine_id or "").strip().lower())
    return dict(definition) if definition is not None else None


def engine_ids(role: str | None = None) -> set[str]:
    wanted = str(role or "").strip().lower()
    return {
        engine_id
        for engine_id, definition in _ENGINE_BY_ID.items()
        if not wanted or str(definition.get("role", "")).strip().lower() == wanted
    }


def engine_role(engine_id: object) -> str | None:
    definition = _ENGINE_BY_ID.get(str(engine_id or "").strip().lower())
    return str(definition.get("role")) if definition is not None and definition.get("role") else None


def engine_label(engine_id: object) -> str:
    text = str(engine_id or "").strip()
    definition = _ENGINE_BY_ID.get(text.lower())
    if definition is not None and definition.get("label"):
        return str(definition["label"])
    return text.upper() if text else "Engine"


def engine_is_generative(engine_id: object) -> bool:
    definition = _ENGINE_BY_ID.get(str(engine_id or "").strip().lower())
    return bool(definition and definition.get("generative", False))


def _app_all_module():
    module = sys.modules.get("app_all")
    if module is not None:
        return module

    # `python app_all.py` registers the running application as __main__, not
    # `app_all`. Most local deployments use exactly that form.
    module = sys.modules.get("__main__")
    filename = Path(str(getattr(module, "__file__", ""))).name.lower() if module else ""
    return module if filename == "app_all.py" else None


def catalog_from_status(state: dict[str, Any]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for definition in ENGINE_REGISTRY:
        engine_id = str(definition["id"])
        runtime = state.get(engine_id)
        if not isinstance(runtime, dict):
            runtime = {}
        entry = dict(definition)
        entry["ready"] = bool(runtime.get("ready", False))
        if runtime.get("setup"):
            entry["setup"] = str(runtime["setup"])
        if runtime.get("error"):
            entry["error"] = str(runtime["error"])
        catalog.append(entry)
    catalog.sort(
        key=lambda item: (
            str(item.get("role", "")),
            int(item.get("order", 0)),
            str(item.get("label", "")),
        )
    )
    return catalog


def decorate_status(state: dict[str, Any]) -> dict[str, Any]:
    value = dict(state)
    value["engines"] = catalog_from_status(value)
    return value


def install_status_endpoint() -> None:
    """Decorate app_all's /engine-status payload with the central engine catalog."""
    app_all = _app_all_module()
    if app_all is None or not hasattr(app_all, "engine_status"):
        return

    original = app_all.engine_status
    if getattr(original, "__engine_catalog_wrapped__", False):
        return

    def engine_status(legacy=None):
        return decorate_status(original(legacy))

    engine_status.__engine_catalog_wrapped__ = True
    app_all.engine_status = engine_status
