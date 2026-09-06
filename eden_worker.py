#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from worker_frame_common import release_cuda, write_generated_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="EDEN midpoint generator worker")
    parser.add_argument("--eden-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    eden_dir = args.eden_dir.resolve()
    sys.path.insert(0, str(eden_dir))

    import torch
    import yaml
    from src.models import load_model
    from src.transport import Sampler, create_transport
    from src.utils import InputPadder

    config = yaml.safe_load((eden_dir / "configs" / "eval_eden.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(config["model_name"], **config["model_args"])
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu")
    model.load_state_dict(checkpoint["eden"])
    del checkpoint
    model.to(device).eval()
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)

    def infer_once(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        frame0 = torch.from_numpy(left.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        frame1 = torch.from_numpy(right.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        h, w = frame0.shape[2:]
        padder = InputPadder([h, w])
        difference = ((torch.mean(torch.cosine_similarity(frame0, frame1), dim=[1, 2])
                       - float(config["cos_sim_mean"])) / float(config["cos_sim_std"])).unsqueeze(1).to(device)
        cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
        new_h, new_w = cond_frames.shape[2:]
        noise = torch.randn([1, new_h // 32 * new_w // 32, config["model_args"]["latent_dim"]], device=device)
        with torch.inference_mode():
            samples = sample_fn(noise, model.denoise, cond_frames=cond_frames, difference=difference)[-1]
            denoise_latents = samples / float(config["vae_scaler"]) + float(config["vae_shift"])
            generated = padder.unpad(model.decode(denoise_latents).clamp(0.0, 1.0))
        result = generated[0, :, :h, :w].mul(255).byte().cpu().numpy().transpose(1, 2, 0)
        del generated, denoise_latents, samples, noise, cond_frames, difference, frame1, frame0
        release_cuda(torch)
        return result

    def infer_rgb(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        original_h, original_w = left.shape[:2]
        last_error = None
        for max_side in (None, 768, 512, 384):
            if max_side is None or max(original_h, original_w) <= max_side:
                l_arr, r_arr = left, right
            else:
                scale = max_side / max(original_h, original_w)
                size = (max(1, round(original_w * scale)), max(1, round(original_h * scale)))
                l_arr = np.asarray(Image.fromarray(left, "RGB").resize(size, Image.Resampling.LANCZOS))
                r_arr = np.asarray(Image.fromarray(right, "RGB").resize(size, Image.Resampling.LANCZOS))
            try:
                pred = infer_once(l_arr, r_arr)
                if pred.shape[:2] != (original_h, original_w):
                    pred = np.asarray(Image.fromarray(pred, "RGB").resize((original_w, original_h), Image.Resampling.LANCZOS))
                return pred
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_error = exc
                release_cuda(torch)
        if last_error:
            raise last_error
        raise RuntimeError("EDEN inference failed.")

    write_generated_sequence(args.input_dir.resolve(), args.output_dir.resolve(), infer_rgb)
    del model
    release_cuda(torch)


if __name__ == "__main__":
    main()
