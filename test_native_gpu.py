import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from native_gpu import peer_memory, worker_device_masks


class NativeGpuTests(unittest.TestCase):
    def test_single_gap_exposes_peers_only_for_supported_engines(self):
        self.assertEqual(worker_device_masks("tooncrafter", ["4", "1"], 1), ["4,1"])
        self.assertEqual(worker_device_masks("resshift", ["4", "1"], 1), ["4"])
        self.assertEqual(worker_device_masks("mog_ani", ["4", "1"], 1, strategy="parallel"), ["4"])

    def test_multiple_gaps_preserve_parallel_dispatch(self):
        self.assertEqual(worker_device_masks("mog_real", ["8", "3", "2"], 4), ["8", "3", "2"])
        self.assertEqual(worker_device_masks("mog_real", ["8", "3", "2"], 4, 1), ["8"])
        self.assertEqual(worker_device_masks("mog_real", ["8", "3", "2"], 4, strategy="peer"), ["8,3,2"])

    def test_no_devices_and_invalid_strategy(self):
        self.assertEqual(worker_device_masks("mog_real", [], 1), [None])
        with self.assertRaises(ValueError):
            worker_device_masks("mog_real", ["0"], 1, strategy="invalid")

    def test_peer_memory_uses_live_free_bytes_and_mutual_access(self):
        cuda = Mock()
        cuda.device_count.return_value = 4
        cuda.current_device.return_value = 1
        cuda.can_device_access_peer.side_effect = lambda first, second: (first, second) in {
            (1, 0), (0, 1), (1, 2), (2, 1), (1, 3)
        }
        cuda.mem_get_info.side_effect = lambda index: {0: (12345, 43210), 2: (67890, 98765)}[index]
        self.assertEqual(peer_memory(SimpleNamespace(cuda=cuda), SimpleNamespace(index=None)),
                         {0: 12345, 2: 67890})
        cuda.mem_get_info.assert_any_call(0)
        cuda.mem_get_info.assert_any_call(2)


if __name__ == "__main__":
    unittest.main()
