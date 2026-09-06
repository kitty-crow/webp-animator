from __future__ import annotations

import unittest

from PIL import Image

import smart_frames
from smart_frames import gap_scores, non_adjacent_removals, reduction_scores


class SmartFrameTests(unittest.TestCase):
    def test_temporal_midpoint_is_reduction_candidate(self):
        first = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
        middle = Image.new("RGBA", (32, 32), (128, 128, 128, 255))
        last = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
        scores = reduction_scores([first, middle, last])
        self.assertEqual(len(scores), 1)
        self.assertLess(scores[0], 1.0)
        self.assertEqual(non_adjacent_removals(scores, 1.0), [1])

    def test_reduction_cache_only_rescores_changed_neighbourhoods(self):
        frames = [
            Image.new("RGBA", (12, 12), (value, value, value, 255))
            for value in (0, 50, 100, 150, 200)
        ]
        with smart_frames._REDUCTION_CACHE_LOCK:
            smart_frames._REDUCTION_CACHE.clear()

        original = smart_frames.reduction_triplet_scores
        calls = []

        def counted(sequence, triplets, *, alpha_threshold=8):
            calls.append(len(triplets))
            return original(sequence, triplets, alpha_threshold=alpha_threshold)

        smart_frames.reduction_triplet_scores = counted
        try:
            reduction_scores(frames)
            self.assertEqual(calls[-1], 3)

            # Remove frame 2 (index 1). Of the two surviving triplets, the latter
            # [old 2,3,4] is unchanged and should be reused from the cache.
            reduced = [frames[0], frames[2], frames[3], frames[4]]
            reduction_scores(reduced)
            self.assertEqual(calls[-1], 1)
        finally:
            smart_frames.reduction_triplet_scores = original
            with smart_frames._REDUCTION_CACHE_LOCK:
                smart_frames._REDUCTION_CACHE.clear()

    def test_large_visual_jump_is_detected_as_missing_information(self):
        first = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
        second = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
        score = gap_scores([first, second])[0]
        self.assertGreater(score, 50.0)

    def test_hidden_rgb_under_zero_alpha_is_not_visual_change(self):
        first = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
        second = Image.new("RGBA", (32, 32), (0, 255, 255, 0))
        self.assertEqual(gap_scores([first, second])[0], 0.0)

    def test_alpha_change_is_visual_change_even_when_rgb_matches(self):
        first = Image.new("RGBA", (32, 32), (120, 80, 30, 0))
        second = Image.new("RGBA", (32, 32), (120, 80, 30, 255))
        self.assertGreater(gap_scores([first, second])[0], 20.0)

    def test_non_adjacent_removal_round_never_deletes_neighbours_together(self):
        chosen = non_adjacent_removals([0.1, 0.2, 0.3, 0.1], 1.0)
        self.assertTrue(chosen)
        for left, right in zip(chosen, chosen[1:]):
            self.assertGreater(right - left, 1)

    def test_protected_frame_is_not_removed(self):
        chosen = non_adjacent_removals([0.1, 0.1, 0.1], 1.0, protected_indexes={2})
        self.assertNotIn(2, chosen)


if __name__ == "__main__":
    unittest.main()
