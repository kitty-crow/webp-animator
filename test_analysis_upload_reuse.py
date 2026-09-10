from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import app_all


class AnalysisUploadReuseTests(unittest.TestCase):
    def test_uploaded_and_stored_source_hashes_are_identical(self):
        uploads = [("a.png", b"alpha"), ("b.png", b"beta"), ("c.png", b"gamma")]
        expected = [hashlib.sha256(payload).hexdigest() for _, payload in uploads]
        self.assertEqual(app_all._sha256_payloads(uploads), expected)

        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, (_, payload) in enumerate(uploads):
                path = Path(temporary) / f"{index}.bin"
                path.write_bytes(payload)
                paths.append(path)
            self.assertEqual(app_all._sha256_paths(paths), expected)

    def test_browser_reuse_patch_supports_insecure_lan_without_webcrypto(self):
        script = bytes(app_all.ADVANCED_SCRIPT)
        self.assertIn(b"analysis-upload-reuse-ui-v2", script)
        self.assertIn(b"sha256Fallback", script)
        self.assertIn(b'data.delete("frames")', script)
        self.assertIn(b'reuse_analysis_sources', script)


if __name__ == "__main__":
    unittest.main()
