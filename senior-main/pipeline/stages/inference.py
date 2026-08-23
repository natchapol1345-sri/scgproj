"""Stage 1 — Load VGGT model and run inference to produce a predictions dict."""
import glob
import os
import sys
import time

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.device import is_mps, autocast_on, aggressive_cleanup

from pipeline.config import DEFAULT_MAX_FRAMES_MPS, IMAGE_EXTENSIONS, VGGT_MODEL_URL


def _select_frames(image_names, max_frames):
    """Uniformly subsample frames if there are more than max_frames, keeping first and last."""
    n = len(image_names)
    if max_frames is None or n <= max_frames:
        return image_names
    indices = np.linspace(0, n - 1, max_frames, dtype=int)
    indices = sorted(set(indices))
    return [image_names[i] for i in indices]


def run_inference(image_folder, device, max_frames=None):
    """Load model, run inference, return predictions dict (numpy) and timings."""
    t0 = time.time()
    print("=" * 60)
    print("STAGE 1: Loading model and running inference")
    print("=" * 60)

    aggressive_cleanup(device)

    # Auto-limit frames on MPS to avoid OOM
    # 9 frames @ 518×518 → global attention over 12k tokens → ~4.9GB just for attn scores
    # 7 frames → ~3.0GB → fits in 30GB MPS with model + activations
    if max_frames is None and is_mps(device):
        max_frames = DEFAULT_MAX_FRAMES_MPS
        print(f"  MPS detected: auto-limiting to {max_frames} frames (override with --max_frames)")

    print("  Loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_MODEL_URL, map_location="cpu"))
    model.eval()
    model = model.to(device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_names = [p for p in image_names if p.lower().endswith(IMAGE_EXTENSIONS)]
    if not image_names:
        print(f"ERROR: No images found in {image_folder}")
        sys.exit(1)

    original_count = len(image_names)
    image_names = _select_frames(image_names, max_frames)
    if len(image_names) < original_count:
        print(f"  Found {original_count} images → selected {len(image_names)} (uniformly spaced)")
    else:
        print(f"  Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"  Preprocessed shape: {images.shape}")

    t1 = time.time()
    print("  Running inference...")
    with torch.no_grad():
        with autocast_on(device):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert all tensors to numpy
    for key in list(predictions.keys()):
        v = predictions[key]
        if isinstance(v, torch.Tensor):
            predictions[key] = v.cpu().float().numpy().squeeze(0)
        elif isinstance(v, list):
            predictions[key] = None
    predictions["pose_enc_list"] = None

    depth_map = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )

    inference_time = time.time() - t1
    print(f"  Inference done in {inference_time:.1f}s")

    del model
    aggressive_cleanup(device)

    return predictions, inference_time
