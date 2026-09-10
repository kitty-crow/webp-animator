from __future__ import annotations

from PIL import Image

import engine_progress


def install(advanced_pipeline_module):
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
    import job_control
    import live_timeline
    import openai_repair_http
    import operation_pipeline
    import pipeline_settings
    import temporal_repair
    import temporal_v2
    import timeline_pipeline

    # The ordered pipeline owns the temporal pass sequence. timeline_pipeline wraps
    # each pass so a canonical current-animation timeline is published, while
    # live_timeline merges in-progress worker output into the right slots and makes
    # repair engines replace repaired slots instead of appending duplicate frame cards.
    operation_pipeline.install(temporal_v2)
    timeline_pipeline.install(operation_pipeline)
    pipeline_settings.install(temporal_v2)
    temporal_repair.install(temporal_v2)
    job_control.install(legacy, advanced_pipeline_module, temporal_v2)
    live_timeline.install(job_control)
    openai_repair_http.install()
