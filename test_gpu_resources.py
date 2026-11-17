from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from gpu_resources import choose_peer, diffusion_target_ladder, worker_device_groups


class GpuResourceTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"WEBP_PEER_OFFLOAD": "auto"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_scheduler_preserves_single_gpu_and_parallel_tasks(self):
        self.assertEqual(worker_device_groups(["0"], 1, allow_peer=True), ["0"])
        self.assertEqual(worker_device_groups(["0", "1"], 2, allow_peer=True), ["0", "1"])
        self.assertEqual(worker_device_groups([], 1), [None])
        self.assertEqual(worker_device_groups(["0", "1"], 0), [])

    def test_spare_peer_group_honours_visibility_and_cap(self):
        self.assertEqual(worker_device_groups(["3", "1"], 1, allow_peer=True), ["3,1"])
        self.assertEqual(worker_device_groups(["3", "1"], 1, 1, allow_peer=True), ["3"])
        self.assertEqual(worker_device_groups(["0", "1"], 1), ["0"])
        self.assertEqual(worker_device_groups(["0", "1", "2", "3"], 2, allow_peer=True),
                         ["0,2", "1,3"])

    def test_peer_requires_mutual_access_and_capacity(self):
        gib = 1024**3
        cuda = SimpleNamespace(
            current_device=lambda: 0,
            device_count=lambda: 3,
            can_device_access_peer=lambda a, b: {a, b} == {0, 2},
            mem_get_info=lambda i: (8 * gib, 11 * gib),
        )
        torch = SimpleNamespace(device=lambda _d: SimpleNamespace(index=None), cuda=cuda)
        self.assertEqual(choose_peer(torch, "cuda", 6 * gib), 2)
        self.assertIsNone(choose_peer(torch, "cuda", 8 * gib))
        with mock.patch.dict(os.environ, {"WEBP_PEER_OFFLOAD": "0"}):
            self.assertIsNone(choose_peer(torch, "cuda", 6 * gib))
            self.assertEqual(worker_device_groups(["0", "1"], 1, allow_peer=True), ["0"])

    def test_quality_preserves_small_gpu_fallbacks(self):
        gib = 1024**3
        self.assertEqual(diffusion_target_ladder(4 * gib)[0], (256, 128))
        self.assertEqual(diffusion_target_ladder(6 * gib)[0], (320, 192))
        self.assertEqual(diffusion_target_ladder(11 * gib)[0], (512, 320))
        for size in (4, 6, 11, 24):
            targets = diffusion_target_ladder(size * gib)
            self.assertEqual(targets[-1], (128, 128))
            self.assertTrue(all(w % 64 == 0 and h % 64 == 0 for w, h in targets))


if __name__ == "__main__":
    unittest.main()
