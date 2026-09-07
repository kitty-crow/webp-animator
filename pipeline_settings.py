from __future__ import annotations

PIPELINE_PREFIX = "pipeline:"


def _extract(value) -> tuple[str, str]:
    parts = [part.strip() for part in str(value or "").replace(";", ",").split(",") if part.strip()]
    pipeline = ""
    kept: list[str] = []
    for part in parts:
        if part.lower().startswith(PIPELINE_PREFIX):
            candidate = part[len(PIPELINE_PREFIX):].strip().replace(">", ",").replace("|", ",")
            if candidate:
                pipeline = candidate
            continue
        kept.append(part)
    return pipeline, ",".join(kept)


def install(temporal_v2_module):
    if getattr(temporal_v2_module, "_pipeline_settings_installed", False):
        return
    temporal_v2_module._pipeline_settings_installed = True
    original = temporal_v2_module.process_job

    def process_job(legacy, job_id, paths, settings):
        value = dict(settings or {})
        pipeline, cleaned_targets = _extract(value.get("target_gaps", ""))
        if pipeline:
            value["temporal_pipeline"] = pipeline
            value["target_gaps"] = cleaned_targets
        return original(legacy, job_id, paths, value)

    temporal_v2_module.process_job = process_job
