from __future__ import annotations


def install(job_control_module) -> None:
    if getattr(job_control_module, "_pipeline_live_installed", False):
        return
    job_control_module._pipeline_live_installed = True
    original = job_control_module._candidate_live_frame

    def candidate(relative):
        if original(relative):
            return True
        parts = [part.lower() for part in relative.parts]
        return any(
            "generator" in part
            or "interpolator" in part
            or "temporal-repair" in part
            for part in parts
        )

    job_control_module._candidate_live_frame = candidate
