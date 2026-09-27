from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from engine_paths import acceleration_status, mog_paths, resshift_paths

ROOT = Path(__file__).resolve().parent
MARKERS = {
    "resshift": "__vfi_resshift__",
    "mog_ani": "__vfi_mog_ani__",
    "mog_real": "__vfi_mog_real__",
}
LABELS = {
    "resshift": "ResShift diffusion",
    "mog_ani": "MoG animation",
    "mog_real": "MoG real-world",
}
_state = threading.local()
_installed = False


UI_PATCH = r'''
/* generative-vfi-ui-v1 */
(() => {
  "use strict";
  const select = document.getElementById("interpolator");
  if (!select || select.dataset.generativeVfi === "1") return;
  select.dataset.generativeVfi = "1";

  const additions = [
    ["resshift", "Multi-Input ResShift Diffusion · generative"],
    ["mog_ani", "MoG · generative animation"],
    ["mog_real", "MoG · generative real-world"],
  ];
  for (const [value, text] of additions) {
    if (select.querySelector(`option[value="${value}"]`)) continue;
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    select.append(option);
  }

  const hint = select.closest("label.option")?.querySelector(".hint");
  const standardHint = hint?.textContent || "";
  const notes = {
    resshift: "Generative residual-diffusion interpolation. Uses both endpoint frames and explicit interpolation time. Slower than RIFE/AMT, but designed for difficult animated in-betweens.",
    mog_ani: "Heavy motion-aware generative interpolation tuned for animation. Extremely slow on old GPUs; the released checkpoint is large and may not fit 4 GB VRAM.",
    mog_real: "Heavy motion-aware generative interpolation tuned for photographic/real-world footage. Extremely slow on old GPUs; the released checkpoint is large and may not fit 4 GB VRAM.",
  };

  function refreshHint() {
    if (!hint) return;
    hint.textContent = notes[select.value] || standardHint;
  }
  select.addEventListener("change", refreshHint);
  refreshHint();

  fetch("/engine-status", { cache: "no-store" })
    .then(response => response.ok ? response.json() : null)
    .then(state => {
      if (!state) return;
      const node = document.getElementById("rifeStatus");
      if (node) {
        const extra = [
          `ResShift: ${state.resshift?.ready ? "ready" : "missing"}`,
          `MoG animation: ${state.mog_ani?.ready ? "ready" : "missing"}`,
          `MoG real: ${state.mog_real?.ready ? "ready" : "missing"}`,
        ];
        if (!node.textContent.includes("ResShift:")) node.textContent += ` · ${extra.join(" · ")}`;
      }
      const vram = Number(state.acceleration?.total_vram?.[0] || 0);
      if (vram > 0 && vram < 6 * 1024 ** 3 && select.value.startsWith("mog_")) {
        if (hint) hint.textContent += " Current GPU has under 6 GB VRAM, so MoG may fail even after the low-resolution fallbacks.";
      }
    })
    .catch(() => {});
})();
'''.strip()


def _marker_engine(value: object) -> str | None:
    text = str(value or "")
    for engine, marker in MARKERS.items():
        if marker in text:
            return engine
    return None


def _append_marker(existing: object, engine: str) -> str:
    marker = MARKERS[engine]
    text = str(existing or "").strip()
    if marker in text:
        return text
    return f"{marker},{text}" if text else marker


def _install_parse_bridge(legacy) -> None:
    original = legacy.parse_multipart
    if getattr(original, "__generative_vfi_wrapped__", False):
        return

    def parse_multipart(content_type: str, body: bytes):
        fields, files = original(content_type, body)
        selected = str(fields.get("interpolator", "")).strip().lower()
        if selected in MARKERS:
            fields["target_gaps"] = _append_marker(fields.get("target_gaps", ""), selected)
            # app_all's historical parser only admits none/rife/amt. Keep AMT as an
            # internal compatibility token; temporal_v2 is intercepted below before
            # any AMT worker is launched.
            fields["interpolator"] = "amt"
        return fields, files

    parse_multipart.__generative_vfi_wrapped__ = True
    legacy.parse_multipart = parse_multipart


def _append_ui_patch() -> None:
    app_all = sys.modules.get("app_all")
    if app_all is None or not hasattr(app_all, "ADVANCED_SCRIPT"):
        return
    marker = b"/* generative-vfi-ui-v1 */"
    if marker in bytes(app_all.ADVANCED_SCRIPT):
        return
    app_all.ADVANCED_SCRIPT = bytes(app_all.ADVANCED_SCRIPT) + b"\n\n" + UI_PATCH.encode("utf-8") + b"\n"


def _visible_gpu_tokens() -> list[str]:
    explicit = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if explicit and explicit not in {"-1", "none", "None"}:
        tokens = [item.strip() for item in explicit.split(",") if item.strip()]
        if tokens:
            return tokens
    status = acceleration_status()
    count = max(0, int(status.get("device_count", 0) or 0))
    return [str(index) for index in range(count)]


def _engine_command(engine: str, manifest: Path, result: Path):
    if engine == "resshift":
        ready, python, source, model = resshift_paths()
        if not ready:
            raise RuntimeError(
                "Multi-Input ResShift Diffusion is selected but is not installed. "
                "Run `python setup_resshift.py`."
            )
        return [
            python,
            ROOT / "resshift_selective_worker.py",
            "--resshift-dir", source,
            "--model-dir", model,
            "--manifest", manifest,
            "--result", result,
        ], source

    if engine in {"mog_ani", "mog_real"}:
        variant = "ani" if engine == "mog_ani" else "real"
        ready, python, source, config, checkpoint, flow_checkpoint = mog_paths(variant)
        if not ready:
            raise RuntimeError(
                f"{LABELS[engine]} is selected but is not installed. "
                f"Run `python setup_mog.py --variant {variant}`."
            )
        return [
            python,
            ROOT / "mog_selective_worker.py",
            "--mog-dir", source,
            "--config", config,
            "--checkpoint", checkpoint,
            "--flow-checkpoint", flow_checkpoint,
            "--variant", variant,
            "--manifest", manifest,
            "--result", result,
        ], source
    raise ValueError(f"Unknown generative VFI engine: {engine}")


def _run_shard(
    engine: str,
    shard_index: int,
    tasks: list[dict],
    worker_dir: Path,
    gpu_token: str | None,
    update: Callable[[int, int, int, str | None], None],
):
    shard_dir = worker_dir / f"shard-{shard_index:02d}"
    output_dir = shard_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = shard_dir / "manifest.json"
    result = shard_dir / "result.json"
    manifest.write_text(
        json.dumps({"output_dir": str(output_dir), "tasks": tasks}, indent=2),
        encoding="utf-8",
    )
    command, cwd = _engine_command(engine, manifest, result)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if gpu_token is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_token)

    process = subprocess.Popen(
        [str(item) for item in command],
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    current = 0
    total = 1
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if line.startswith("PROGRESS "):
            try:
                _, current_text, total_text = line.split()
                current = max(0, int(current_text))
                total = max(1, int(total_text))
                update(shard_index, current, total, None)
            except Exception:
                pass
        elif line:
            tail.append(line)
            tail = tail[-40:]
            # Surface meaningful worker phase/fallback changes while keeping the
            # global percentage controlled by real diffusion-step progress.
            if any(token in line.lower() for token in ("worker", "low-vram", "retry", "oom", "device=")):
                update(shard_index, current, total, line)

    code = process.wait()
    if code != 0:
        details = "\n".join(tail[-14:]) or "No worker diagnostics."
        raise RuntimeError(f"{LABELS[engine]} worker failed:\n{details}")
    value = json.loads(result.read_text(encoding="utf-8"))
    return list(value.get("tasks", []))


def run_generative_interpolator(
    engine: str,
    tasks: list[dict],
    stage_dir: Path,
    progress: Callable[[float, str], None] | None = None,
):
    if not tasks:
        return {}
    worker_dir = stage_dir / f"{engine}-interpolator-v2"
    worker_dir.mkdir(parents=True, exist_ok=True)

    gpu_tokens = _visible_gpu_tokens()
    requested = int(os.environ.get("GENERATIVE_VFI_GPU_WORKERS", "0") or 0)
    if requested > 0:
        gpu_tokens = gpu_tokens[:requested]
    if not gpu_tokens:
        gpu_tokens = [None]
    worker_count = min(len(tasks), len(gpu_tokens))
    worker_count = max(1, worker_count)

    shards: list[list[dict]] = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        shards[index % worker_count].append(task)

    lock = threading.Lock()
    state = {index: (0, 1) for index in range(worker_count)}

    def update(index: int, current: int, total: int, detail: str | None):
        with lock:
            state[index] = (max(0, current), max(1, total))
            fraction = sum(min(1.0, c / t) for c, t in state.values()) / worker_count
            completed = sum(c for c, _ in state.values())
            aggregate_total = sum(t for _, t in state.values())
        if progress:
            suffix = f" · {worker_count} GPU worker{'s' if worker_count != 1 else ''}" if worker_count > 1 else ""
            if detail:
                message = f"{LABELS[engine]} · {detail}{suffix}"
            else:
                message = f"{LABELS[engine]} diffusion {completed}/{aggregate_total}{suffix}"
            progress(fraction, message)

    outputs = []
    if worker_count == 1:
        outputs.extend(
            _run_shard(engine, 0, shards[0], worker_dir, gpu_tokens[0], update)
        )
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"{engine}-gpu") as pool:
            futures = {
                pool.submit(
                    _run_shard,
                    engine,
                    index,
                    shard,
                    worker_dir,
                    gpu_tokens[index],
                    update,
                ): index
                for index, shard in enumerate(shards)
                if shard
            }
            for future in as_completed(futures):
                outputs.extend(future.result())

    if progress:
        progress(1.0, f"{LABELS[engine]} interpolation complete")
    return {str(item.get("id")): dict(item) for item in outputs}


def _prepare_generative_results(
    legacy,
    temporal_v2,
    engine: str,
    plans,
    records,
    stage_dir: Path,
    settings: dict,
    progress: Callable[[int, str], None],
):
    if not plans:
        return {}, []

    source_paths = temporal_v2.base._save_records(records, stage_dir / "temporal-input")
    generator = str(settings.get("frame_generator", "none")).lower()
    manual_normal, manual_loop = temporal_v2.parse_target_gap_spec(settings.get("target_gaps", ""))
    using_gap_fill = bool(settings.get("smart_missing", False) or manual_normal or manual_loop)
    fill_count = temporal_v2._requested_fill_count(settings, using_gap_fill)
    threshold = float(settings.get("missing_threshold", 12.0))
    alpha_threshold = int(settings.get("alpha_threshold", 8))

    endpoint_paths = {}
    for ordinal, plan in enumerate(plans):
        key = f"gap-{ordinal}"
        if plan.kind == "loop":
            endpoint_paths[key] = (source_paths[-1], source_paths[0])
        else:
            assert plan.record_index is not None
            endpoint_paths[key] = (
                source_paths[plan.record_index],
                source_paths[plan.record_index + 1],
            )

    active_keys = set(endpoint_paths)
    if fill_count == 0:
        frames = [record.image for record in records]
        for ordinal, plan in enumerate(plans):
            key = f"gap-{ordinal}"
            if plan.score is not None:
                score = plan.score
            elif plan.kind == "loop":
                score = temporal_v2.pair_scores(
                    frames,
                    [(len(frames) - 1, 0)],
                    alpha_threshold=alpha_threshold,
                )[0]
            else:
                assert plan.record_index is not None
                score = temporal_v2.pair_scores(
                    frames,
                    [(plan.record_index, plan.record_index + 1)],
                    alpha_threshold=alpha_threshold,
                )[0]
            if score <= threshold:
                active_keys.discard(key)

    final_refs = {key: [] for key in endpoint_paths}
    diagnostics = []
    if not active_keys:
        return final_refs, diagnostics

    if generator not in {"eden", "speed"}:
        tasks = []
        for key in sorted(active_keys):
            left, right = endpoint_paths[key]
            tasks.append({
                "id": key,
                "left": str(left),
                "right": str(right),
                "count": fill_count,
                "threshold": threshold,
                "max_frames": temporal_v2.AUTO_FILL_MAX_FRAMES,
                "alpha_threshold": alpha_threshold,
            })
        progress(56, f"Starting {LABELS[engine]} gap filling")
        result = run_generative_interpolator(
            engine,
            tasks,
            stage_dir,
            progress=lambda fraction, message: progress(56 + round(fraction * 29), message),
        )
        for key, item in result.items():
            final_refs[key] = temporal_v2._task_refs(item)
            if item.get("limit_reached"):
                diagnostics.append({
                    "gap": key,
                    "engine": engine,
                    "limit_reached": True,
                    "max_score": item.get("max_score"),
                })
        return final_refs, diagnostics

    # Preserve the established EDEN/SPEED structural-anchor pipeline. Only the
    # interpolation stage changes to the selected generative model.
    generator_tasks = []
    for key in sorted(active_keys):
        left, right = endpoint_paths[key]
        generator_tasks.append({
            "id": key,
            "left": str(left),
            "right": str(right),
            "count": 1,
            "alpha_threshold": alpha_threshold,
        })
    progress(48, f"Starting {generator.upper()} structural midpoint generation")
    generated = temporal_v2._run_generator(
        generator,
        generator_tasks,
        stage_dir,
        progress=lambda fraction: progress(
            48 + round(fraction * 17),
            f"{generator.upper()} structural midpoint generation",
        ),
    )

    interp_tasks = []
    mapping = {}
    for key in sorted(active_keys):
        anchor_refs = temporal_v2._task_refs(generated.get(key, {}))
        if not anchor_refs:
            continue
        anchor = min(anchor_refs, key=lambda ref: abs(float(ref["t"]) - 0.5))
        anchor["t"] = 0.5
        left, right = endpoint_paths[key]

        if fill_count > 0:
            remaining = max(0, fill_count - 1)
            left_count = (remaining + 1) // 2
            right_count = remaining - left_count
        else:
            left_count = right_count = 0

        keep_anchor = fill_count == 0 or fill_count % 2 == 1
        if keep_anchor:
            final_refs[key].append(anchor)
        if fill_count > 0 and not keep_anchor:
            left_count = right_count = fill_count // 2

        if left_count > 0 or fill_count == 0:
            task_id = f"{key}-left"
            interp_tasks.append({
                "id": task_id,
                "left": str(left),
                "right": anchor["path"],
                "count": left_count if fill_count > 0 else 0,
                "threshold": threshold,
                "max_frames": max(1, temporal_v2.AUTO_FILL_MAX_FRAMES // 2),
                "alpha_threshold": alpha_threshold,
            })
            mapping[task_id] = (key, 0.0, 0.5)
        if right_count > 0 or fill_count == 0:
            task_id = f"{key}-right"
            interp_tasks.append({
                "id": task_id,
                "left": anchor["path"],
                "right": str(right),
                "count": right_count if fill_count > 0 else 0,
                "threshold": threshold,
                "max_frames": max(1, temporal_v2.AUTO_FILL_MAX_FRAMES // 2),
                "alpha_threshold": alpha_threshold,
            })
            mapping[task_id] = (key, 0.5, 1.0)

    if interp_tasks:
        progress(66, f"Starting {LABELS[engine]} anchored interpolation")
        interpolated = run_generative_interpolator(
            engine,
            interp_tasks,
            stage_dir,
            progress=lambda fraction, message: progress(66 + round(fraction * 19), message),
        )
        for task_id, item in interpolated.items():
            if task_id not in mapping:
                continue
            key, start, end = mapping[task_id]
            for ref in temporal_v2._task_refs(item):
                local_t = float(ref["t"])
                final_refs[key].append({
                    "path": ref["path"],
                    "t": start + (end - start) * local_t,
                })
            if item.get("limit_reached"):
                diagnostics.append({
                    "gap": key,
                    "engine": engine,
                    "limit_reached": True,
                    "max_score": item.get("max_score"),
                })

    for key in final_refs:
        final_refs[key].sort(key=lambda ref: float(ref["t"]))
        if fill_count > 0 and len(final_refs[key]) > fill_count:
            final_refs[key] = final_refs[key][:fill_count]
    return final_refs, diagnostics


def _install_temporal_bridge(legacy, temporal_v2) -> None:
    original_prepare = temporal_v2._prepare_engine_results
    original_insert = temporal_v2._insert_gap_frames
    if getattr(original_prepare, "__generative_vfi_wrapped__", False):
        return

    def prepare_engine_results(legacy_arg, plans, records, stage_dir, settings, progress):
        engine = _marker_engine(settings.get("target_gaps", ""))
        _state.engine = engine
        if engine is None:
            return original_prepare(legacy_arg, plans, records, stage_dir, settings, progress)
        return _prepare_generative_results(
            legacy_arg,
            temporal_v2,
            engine,
            plans,
            records,
            stage_dir,
            settings,
            progress,
        )

    def insert_gap_frames(records, plan, refs, *, engine_label, source_count):
        engine = getattr(_state, "engine", None)
        if engine:
            parts = [part for part in str(engine_label).split("+") if part]
            parts = [engine if part == "amt" else part for part in parts]
            engine_label = "+".join(parts) or engine
        return original_insert(
            records,
            plan,
            refs,
            engine_label=engine_label,
            source_count=source_count,
        )

    prepare_engine_results.__generative_vfi_wrapped__ = True
    temporal_v2._prepare_engine_results = prepare_engine_results
    temporal_v2._insert_gap_frames = insert_gap_frames


def install(legacy, temporal_v2) -> None:
    global _installed
    if _installed:
        return
    _installed = True
    _install_parse_bridge(legacy)
    _install_temporal_bridge(legacy, temporal_v2)
    _append_ui_patch()
