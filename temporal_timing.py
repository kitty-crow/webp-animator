from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

import engine_progress


def _publish_running_app_alias() -> None:
    """Make patch modules see the direct `python app_all.py` process as app_all.

    Several feature modules were originally written for `import app_all` and looked
    only in sys.modules["app_all"]. Direct script execution registers the same module
    as __main__, which made UI/runtime patches silently disappear depending on how the
    server was launched. Publish one canonical alias before installing any patches so
    every module observes the same live application object and no duplicate import is
    created.
    """
    if sys.modules.get("app_all") is not None:
        return
    main = sys.modules.get("__main__")
    filename = Path(str(getattr(main, "__file__", ""))).name.lower() if main else ""
    if filename == "app_all.py":
        sys.modules["app_all"] = main


def install(advanced_pipeline_module):
    _publish_running_app_alias()
    FrameRecord = advanced_pipeline_module.FrameRecord

    def insert_interpolator_results(records, task_specs, results, engine):
        for task_id, record_index, left_source, depth in reversed(task_specs):
            refs = list(results.get(task_id, []))
            if not refs:
                continue

            normalised = []
            for offset, ref in enumerate(refs):
                if isinstance(ref, dict):
                    path = ref.get("path")
                    t = float(ref.get("t", (offset + 1) / (len(refs) + 1)))
                else:
                    path = ref
                    t = (offset + 1) / (len(refs) + 1)
                if path:
                    normalised.append((max(0.0, min(1.0, t)), str(path)))
            normalised.sort(key=lambda item: item[0])
            if not normalised:
                continue

            left_record = records[record_index]
            old_duration = float(left_record.duration)
            times = [0.0] + [item[0] for item in normalised] + [1.0]
            intervals = [
                max(0.0, old_duration * (times[index + 1] - times[index]))
                for index in range(len(times) - 1)
            ]
            left_record.duration = intervals[0]

            inserted = []
            for offset, (_, path) in enumerate(normalised):
                with Image.open(path) as image:
                    middle = image.convert("RGBA").copy()
                inserted.append(
                    FrameRecord(
                        image=middle,
                        duration=intervals[offset + 1],
                        source_indices=set(),
                        generated=True,
                        engine=engine,
                        between_source_frames=(left_source, left_source + 1),
                        recursive_depth=depth,
                    )
                )
            records[record_index + 1:record_index + 1] = inserted

    advanced_pipeline_module._insert_interpolator_results = insert_interpolator_results
    engine_progress.install(advanced_pipeline_module)

    import app as legacy
    import analysis_upload_reuse
    import engine_catalog
    import generative_models_ui
    import generative_pipeline_bridge
    import generative_process_control
    import generative_vfi
    import generative_vfi_restore
    import geometry_cache
    import job_control
    import live_timeline
    import looped_animation
    import operation_pipeline
    import pipeline_settings
    import rife_compat
    import temporal_repair
    import temporal_v2
    import timeline_pipeline
    import tooncrafter_vfi

    rife_compat.install(legacy)
    engine_catalog.install_status_endpoint()
    operation_pipeline.install(temporal_v2)
    looped_animation.install(temporal_v2)
    timeline_pipeline.install(operation_pipeline)
    pipeline_settings.install(temporal_v2)
    temporal_repair.install(temporal_v2)
    # Analysis and render both enter _normalise_geometry through this wrapper. The
    # analysis result is stored outside its temporary snapshot, so Generate WebP can
    # reuse the exact matched/canvas-normalised frames when sources/settings match.
    geometry_cache.install(advanced_pipeline_module, temporal_v2)
    analysis_upload_reuse.install_ui_patch()
    tooncrafter_vfi.install_backend()

    # The engine catalogue is now the sole browser-side engine-option source. The
    # original generative_vfi UI patch manually appended ResShift/MoG options and can
    # race the catalogue population, so keep only its backend bridge behaviour.
    generative_vfi._append_ui_patch = lambda: None
    generative_vfi.install(legacy, temporal_v2)
    generative_pipeline_bridge.install(temporal_v2)
    generative_vfi_restore.install_ui_patch()
    generative_models_ui.install_ui_patch()
    job_control.install(legacy, advanced_pipeline_module, temporal_v2)
    generative_process_control.install(generative_vfi)
    live_timeline.install(job_control)
