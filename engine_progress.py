from __future__ import annotations

import threading


_state = threading.local()
_installed = False
_ENGINE_NOUN = {
    "EDEN": "frame",
    "SPEED": "frame",
    "RIFE": "step",
    "AMT": "step",
}


def install(advanced_pipeline_module) -> None:
    """Preserve worker x/y progress in the persistent job status.

    Every temporal worker prints ``PROGRESS current total``. The temporal pipeline
    still uses the derived fraction for the main percentage bar, while this adapter
    keeps the concrete count visible in the status message. Generator engines use
    ``frame X/Y``; interpolators use ``step X/Y``.
    """
    global _installed
    if _installed:
        return
    _installed = True

    import app as legacy

    original_run_process = advanced_pipeline_module._run_process
    original_set_job = legacy.set_job

    def run_process(command, *, cwd, progress_callback=None, label: str):
        engine = str(label).upper()
        if progress_callback is None or engine not in _ENGINE_NOUN:
            return original_run_process(
                command,
                cwd=cwd,
                progress_callback=progress_callback,
                label=label,
            )

        def counted_progress(current, total):
            _state.engine = engine
            _state.current = max(0, int(current))
            _state.total = max(1, int(total))
            try:
                return progress_callback(current, total)
            finally:
                _state.engine = None
                _state.current = None
                _state.total = None

        return original_run_process(
            command,
            cwd=cwd,
            progress_callback=counted_progress,
            label=label,
        )

    def set_job(job_id: str, **changes):
        engine = getattr(_state, "engine", None)
        current = getattr(_state, "current", None)
        total = getattr(_state, "total", None)
        message = changes.get("message")
        noun = _ENGINE_NOUN.get(engine)
        if (
            noun
            and current is not None
            and total is not None
            and isinstance(message, str)
            and engine in message.upper()
            and f"{noun.upper()} " not in message.upper()
        ):
            changes = dict(changes)
            changes["message"] = f"{message} · {noun} {current}/{total}"
        return original_set_job(job_id, **changes)

    advanced_pipeline_module._run_process = run_process
    legacy.set_job = set_job
