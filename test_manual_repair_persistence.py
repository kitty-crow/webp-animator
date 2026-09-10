from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from global_jobs import GlobalJobStore
import openai_repair_http


class ManualRepairPersistenceTests(unittest.TestCase):
    def test_live_manual_repair_replaces_timeline_frame_and_rebuilds_webp(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = GlobalJobStore(Path(temporary))
            job_id = "a" * 32
            store.ensure(job_id)
            store.update(job_id, status="done", settings={"duration": 45, "quality": 90, "lossy": False})
            job_root = store.job_dir(job_id)
            advanced = job_root / "advanced"
            revision = advanced / "live-timeline" / "rev-001"
            revision.mkdir(parents=True)

            frames = []
            for index, colour in enumerate(((20, 40, 60, 255), (70, 90, 110, 255), (120, 140, 160, 255))):
                path = revision / f"{index:06d}.png"
                Image.new("RGBA", (16, 16), colour).save(path)
                frames.append(
                    {
                        "key": f"timeline:{index}",
                        "name": f"frame-{index + 1:04d}.png",
                        "rel": path.relative_to(advanced).as_posix(),
                        "stage": "Generated · SPEED" if index == 1 else "Original",
                        "generated": index == 1,
                        "engine": "speed" if index == 1 else "",
                        "timeline_index": index,
                        "duration": 45.0,
                    }
                )
            manifest_path = advanced / "live-timeline" / "manifest.json"
            manifest_path.write_text(json.dumps({"completed_pass": 1, "frames": frames}), encoding="utf-8")

            repaired = Image.new("RGBA", (16, 16), (230, 20, 30, 255))
            with mock.patch.dict(sys.modules, {"app_all": SimpleNamespace(GLOBAL=store)}):
                result = openai_repair_http._persist_finished_repair(job_id, 1, repaired)

            self.assertTrue(result["persisted"])
            self.assertTrue(result["webp_rebuilt"])
            self.assertTrue((job_root / "animation.webp").is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            middle = manifest["frames"][1]
            self.assertTrue(middle["manual_repair"])
            self.assertIn("live-timeline/manual-repairs/", middle["rel"])
            self.assertIn("openai-repair", middle["engine"])
            with Image.open(advanced / middle["rel"]) as image:
                self.assertEqual(image.convert("RGBA").getpixel((0, 0)), (230, 20, 30, 255))
            with Image.open(job_root / "animation.webp") as animation:
                self.assertEqual(getattr(animation, "n_frames", 1), 3)


if __name__ == "__main__":
    unittest.main()
