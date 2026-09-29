from __future__ import annotations

import threading

import generative_vfi
import operation_pipeline


_state = threading.local()
_installed = False


def _selected_engine(settings: dict) -> str | None:
    explicit = str(settings.get("interpolator", "")).strip().lower()
    if explicit in generative_vfi.MARKERS:
        return explicit
    return generative_vfi._marker_engine(settings.get("target_gaps", ""))


def install(temporal_v2) -> None:
    """Make the ordered temporal pipeline honour generative interpolators.

    app_all historically only admitted RIFE/AMT, so generative_vfi carries the
    selected engine through parsing as a marker while using AMT as a compatibility
    token. The ordered operation pipeline bypasses _prepare_engine_results and used
    that token literally, which launched AMT. This bridge intercepts the ordered
    interpolation pass and dispatches the actual selected generative worker instead.

    The state is thread-local because persistent jobs may render concurrently.
    """
    global _installed
    if _installed:
        return
    _installed = True

    original_run_interpolator = temporal_v2._run_interpolator
    original_engine_pass = operation_pipeline._engine_pass

    def run_interpolator(legacy, engine, tasks, stage_dir, progress=None):
        selected = getattr(_state, "engine", None)
        if not selected:
            return original_run_interpolator(
                legacy,
                engine,
                tasks,
                stage_dir,
                progress=progress,
            )

        def forward(fraction: float, _message: str) -> None:
            if progress:
                progress(fraction)

        return generative_vfi.run_generative_interpolator(
            selected,
            tasks,
            stage_dir,
            progress=forward if progress else None,
        )

    def engine_pass(
        temporal_v2_module,
        legacy,
        records,
        plans,
        source_count,
        settings,
        stage_dir,
        *,
        operation,
        pass_number,
        pass_total,
        progress,
    ):
        selected = _selected_engine(settings)
        if operation != "interpolate" or not selected:
            return original_engine_pass(
                temporal_v2_module,
                legacy,
                records,
                plans,
                source_count,
                settings,
                stage_dir,
                operation=operation,
                pass_number=pass_number,
                pass_total=pass_total,
                progress=progress,
            )

        label = generative_vfi.LABELS.get(selected, selected)
        prior_ids = {id(record) for record in records}

        def mapped_progress(fraction, message):
            text = str(message)
            # The ordered pipeline still sees the historical AMT compatibility token.
            # Never expose that implementation detail to the user.
            text = text.replace("AMT", label).replace("amt", label)
            progress(fraction, text)

        _state.engine = selected
        try:
            diagnostics = original_engine_pass(
                temporal_v2_module,
                legacy,
                records,
                plans,
                source_count,
                settings,
                stage_dir,
                operation=operation,
                pass_number=pass_number,
                pass_total=pass_total,
                progress=mapped_progress,
            )
        finally:
            _state.engine = None

        # _engine_pass labels inserted frames with the compatibility token. Correct
        # only frames created by this pass so persisted metadata names the real model.
        for record in records:
            if id(record) in prior_ids or not getattr(record, "generated", False):
                continue
            if str(getattr(record, "engine", "")).lower() == "amt":
                record.engine = selected

        corrected = []
        for item in diagnostics:
            value = dict(item)
            if str(value.get("engine", "")).lower() == "amt":
                value["engine"] = selected
            corrected.append(value)
        return corrected

    temporal_v2._run_interpolator = run_interpolator
    operation_pipeline._engine_pass = engine_pass
