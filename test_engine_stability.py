from __future__ import annotations

import os
import unittest
from unittest import mock

import engine_catalog
import generative_pipeline_bridge
import generative_vfi
import model_offload
import setup_engine_common
import tooncrafter_vfi


tooncrafter_vfi.install_backend()


class EngineStabilityTests(unittest.TestCase):
    def test_every_generative_interpolator_is_catalogued(self):
        expected = {"resshift", "mog_ani", "mog_real", "tooncrafter"}
        self.assertTrue(expected.issubset(engine_catalog.engine_ids("interpolator")))
        for engine in expected:
            definition = engine_catalog.engine_definition(engine)
            self.assertIsNotNone(definition)
            self.assertTrue(definition["generative"])
            self.assertTrue(definition["requires_cuda"])

    def test_rife_is_also_runtime_cuda_guarded(self):
        definition = engine_catalog.engine_definition("rife")
        self.assertIsNotNone(definition)
        self.assertTrue(definition["requires_cuda"])

    def test_legacy_marker_resolves_to_real_engine_before_dispatch(self):
        for engine, marker in generative_vfi.MARKERS.items():
            settings = {
                "interpolator": "amt",
                "target_gaps": f"2,{marker},loop",
            }
            self.assertEqual(generative_pipeline_bridge._selected_engine(settings), engine)

    def test_direct_generative_selection_is_also_understood(self):
        for engine in ("resshift", "mog_ani", "mog_real", "tooncrafter"):
            self.assertEqual(
                generative_pipeline_bridge._selected_engine(
                    {"interpolator": engine, "target_gaps": ""}
                ),
                engine,
            )

    def test_cuda11_driver_selects_cu118_for_old_research_stacks(self):
        with mock.patch.object(setup_engine_common, "nvidia_driver_cuda_version", return_value=(11, 6)):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MOG_CUDA_VARIANT", None)
                variant, driver = setup_engine_common.choose_torch_cuda_variant(
                    "MOG_CUDA_VARIANT",
                    cuda12_variant="cu121",
                )
        self.assertEqual(driver, (11, 6))
        self.assertEqual(variant, "cu118")

    def test_cuda12_driver_selects_cuda12_family(self):
        with mock.patch.object(setup_engine_common, "nvidia_driver_cuda_version", return_value=(12, 4)):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TOONCRAFTER_CUDA_VARIANT", None)
                variant, driver = setup_engine_common.choose_torch_cuda_variant(
                    "TOONCRAFTER_CUDA_VARIANT",
                    cuda12_variant="cu121",
                )
        self.assertEqual(driver, (12, 4))
        self.assertEqual(variant, "cu121")

    def test_explicit_cuda_override_wins(self):
        with mock.patch.dict(os.environ, {"MOG_CUDA_VARIANT": "cu118"}, clear=False):
            with mock.patch.object(setup_engine_common, "nvidia_driver_cuda_version", return_value=(12, 4)):
                variant, _ = setup_engine_common.choose_torch_cuda_variant(
                    "MOG_CUDA_VARIANT",
                    cuda12_variant="cu121",
                )
        self.assertEqual(variant, "cu118")

    def test_catalog_hides_cuda_only_engine_when_its_runtime_is_broken(self):
        state = {
            "resshift": {"ready": True, "python": "fake-python"},
            "amt": {"ready": True, "python": "fake-python"},
        }
        with mock.patch.object(
            engine_catalog,
            "_cuda_runtime_probe",
            return_value={"ready": False, "error": "CUDA unavailable"},
        ):
            catalog = engine_catalog.catalog_from_status(state)
        by_id = {entry["id"]: entry for entry in catalog}
        self.assertFalse(by_id["resshift"]["ready"])
        self.assertEqual(by_id["resshift"]["error"], "CUDA unavailable")
        # AMT has a CPU path, so a missing CUDA runtime must not make the engine
        # disappear from the catalogue entirely.
        self.assertTrue(by_id["amt"]["ready"])

    def test_decorated_top_level_readiness_matches_catalogue(self):
        state = {
            "rife": {"ready": True, "python": "fake-python"},
            "amt": {"ready": True, "python": "fake-python"},
        }
        with mock.patch.object(
            engine_catalog,
            "_cuda_runtime_probe",
            return_value={"ready": False, "error": "driver mismatch"},
        ):
            decorated = engine_catalog.decorate_status(state)
        by_id = {entry["id"]: entry for entry in decorated["engines"]}
        self.assertFalse(by_id["rife"]["ready"])
        self.assertFalse(decorated["rife"]["ready"])
        self.assertEqual(decorated["rife"]["error"], "driver mismatch")
        self.assertTrue(decorated["amt"]["ready"])

    def test_meta_offload_guard_prevents_legacy_module_to_from_copying_meta(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable in test environment: {exc}")

        layer = torch.nn.Linear(4, 4, device="meta")
        self.assertTrue(model_offload._module_has_meta_state(layer))
        guarded = model_offload._guard_meta_module_moves(layer)
        self.assertGreaterEqual(guarded, 1)
        # A normal Module.to('cuda') on a meta parameter raises because meta has no
        # backing storage. Once a module is Accelerate-managed, that relocation is
        # redundant: the forward hook streams the real weight to its execution device.
        self.assertIs(layer.to(torch.device("cuda")), layer)
        self.assertEqual(next(layer.parameters()).device.type, "meta")


if __name__ == "__main__":
    unittest.main()
