"""Top-level pipeline orchestrator — runs Stages 1-7 end to end."""
import glob
import os
import shutil
import sys
import time

import numpy as np

from vggt.utils.device import get_device

from pipeline.cli import parse_args
from pipeline.config import IMAGE_EXTENSIONS
from pipeline.utils.runlog import RunLogger
from pipeline.utils.seeding import seed_everything

from pipeline.stages.inference import run_inference
from pipeline.stages.pointcloud import export_ply
from pipeline.stages.clean import clean_and_extract
from pipeline.stages.reconstruct import reconstruct_mesh_stage
from pipeline.stages.watertight import watertight_stage
from pipeline.stages.volume import compute_volumes


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _print_banner(args, device):
    print(f"╔{'═' * 58}╗")
    print(f"║  VGGT Full Pipeline                                      ║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Device        : {device:<40}║")
    print(f"║  Input         : {args.image_folder:<40}║")
    print(f"║  Pred. mode    : {args.prediction_mode:<40}║")
    print(f"║  Conf. thresh  : {args.conf_thres:<40}║")
    print(f"║  Skip mesh     : {str(args.skip_mesh):<40}║")
    recon_label = args.recon_method
    if args.box_recon_method and args.box_recon_method != args.recon_method:
        recon_label += f" (box={args.box_recon_method})"
    if args.obj_recon_method and args.obj_recon_method != args.recon_method:
        recon_label += f" (obj={args.obj_recon_method})"
    print(f"║  Recon method  : {recon_label:<40}║")
    print(f"║  Watertight    : {str(not args.no_watertight):<40}║")
    print(f"║  Seed          : {args.seed:<40}║")
    print(f"║  Leg segment   : {str(args.segment_leg):<40}║")
    print(f"╚{'═' * 58}╝")


def _copy_images_to_target(image_folder, target_images_dir):
    """Copy input images to target/images/ for demo_gradio compatibility."""
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_files = [p for p in image_files if p.lower().endswith(IMAGE_EXTENSIONS)]
    for src in image_files:
        dst = os.path.join(target_images_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def _print_summary(total_time, inference_time, ply_path, scene_recon_path,
                   recon_mesh_paths, scene_wt_path, wt_mesh_paths,
                   npz_path2, target_dir, target_images_dir):
    print()
    print(f"╔{'═' * 58}╗")
    print(f"║  Pipeline Complete                                       ║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Total time    : {total_time:>6.1f}s{' ' * 33}║")
    print(f"║  Inference     : {inference_time:>6.1f}s{' ' * 33}║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Outputs:                                                ║")
    print(f"║    PLY         : {ply_path:<40}║")
    if scene_recon_path:
        print(f"║    Scene recon : {scene_recon_path:<40}║")
    for p in recon_mesh_paths:
        name = os.path.basename(p)
        print(f"║    Recon       : {name:<40}║")
    if scene_wt_path:
        print(f"║    Scene wt    : {scene_wt_path:<40}║")
    for p in wt_mesh_paths:
        name = os.path.basename(p)
        print(f"║    Wt          : {name:<40}║")
    print(f"║    Predictions : {npz_path2:<40}║")
    print(f"║    Target dir  : {target_dir:<40}║")
    print(f"╚{'═' * 58}╝")
    print()
    print("To view results interactively:")
    print(f"  python viewer.py {ply_path}")
    if scene_recon_path:
        print(f"  python viewer.py {scene_recon_path}  # recon scene")
    for p in recon_mesh_paths:
        print(f"  python viewer.py {p}")
    if scene_wt_path:
        print(f"  python viewer.py {scene_wt_path}  # watertight scene")
    for p in wt_mesh_paths:
        print(f"  python viewer.py {p}")
    print()
    print("To use with demo_gradio.py:")
    print(f"  The predictions are saved at: {target_dir}/predictions.npz")
    print(f"  Images are at: {target_images_dir}/")


def main():
    args = parse_args()
    device = get_device()
    total_t0 = time.time()

    seed_everything(args.seed)

    _print_banner(args, device)

    if not os.path.isdir(args.image_folder):
        print(f"\nERROR: Input folder not found: {args.image_folder}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.join(_PROJECT_ROOT, "output")
    os.makedirs(args.output_dir, exist_ok=True)

    logger = None
    inference_time = None
    obj_vol_cm3 = None
    box_vol_cm3 = None
    if args.log:
        logger = RunLogger(os.path.join(_PROJECT_ROOT, "log.csv"), device)
        logger.start()

    try:
        # target_dir mirrors demo_gradio's expected layout (predictions.npz + images/).
        target_dir = os.path.join(args.output_dir, "target")
        target_images_dir = os.path.join(target_dir, "images")
        os.makedirs(target_images_dir, exist_ok=True)
        _copy_images_to_target(args.image_folder, target_images_dir)

        # ── Stage 1: Inference ──
        predictions, inference_time = run_inference(args.image_folder, device, args.max_frames)
        print(f"[DBG-stage] stage1 inference: {inference_time:.2f}s")

        # Save predictions (compatible with demo_gradio)
        npz_path = os.path.join(target_dir, "predictions.npz")
        save_dict = {k: v for k, v in predictions.items() if v is not None}
        np.savez_compressed(npz_path, **save_dict)
        print(f"  Saved predictions: {npz_path}")

        npz_path2 = os.path.join(args.output_dir, "predictions.npz")
        shutil.copy2(npz_path, npz_path2)

        # ── Stage 2: Export PLY ──
        _dbg_t = time.time()
        ply_path = export_ply(predictions, args.output_dir, args)
        print(f"[DBG-stage] stage2 export_ply: {time.time() - _dbg_t:.2f}s")

        # ── Stages 3-5: Clean + Reconstruct + Watertight ──
        scene_recon_path = None
        recon_mesh_paths = []
        scene_wt_path = None
        wt_mesh_paths = []

        if not args.skip_mesh:
            _dbg_t = time.time()
            object_paths = clean_and_extract(
                ply_path, args.output_dir, args.num_objects, seed=args.seed,
                segment_leg=args.segment_leg,
                segment_height_axis=args.segment_height_axis,
                fill_enabled=not args.no_fill)
            print(f"[DBG-stage] stage3 clean_and_extract: {time.time() - _dbg_t:.2f}s")
            if object_paths:
                _dbg_t = time.time()
                scene_recon_path, recon_mesh_paths = reconstruct_mesh_stage(
                    object_paths, args.output_dir, seed=args.seed,
                    method=args.recon_method,
                    box_method=args.box_recon_method,
                    obj_method=args.obj_recon_method)
                print(f"[DBG-stage] stage4 reconstruct: {time.time() - _dbg_t:.2f}s")

                if recon_mesh_paths and not args.no_watertight:
                    _dbg_t = time.time()
                    scene_wt_path, wt_mesh_paths = watertight_stage(
                        recon_mesh_paths, args.output_dir)
                    print(f"[DBG-stage] stage5 watertight: {time.time() - _dbg_t:.2f}s")
        else:
            print("\n  (Skipping mesh stages — --skip_mesh was set)")

        # ── Stage 6: Volumes ──
        vol_objects = wt_mesh_paths or recon_mesh_paths
        if vol_objects:
            _dbg_t = time.time()
            vol_df = compute_volumes(vol_objects,
                                     voxel_res=args.voxel_res,
                                     auto_res=args.auto_res)
            print(f"[DBG-stage] stage6 volumes: {time.time() - _dbg_t:.2f}s")
            if vol_df is not None:
                box_rows = vol_df[vol_df["is_ref"]]
                obj_rows = vol_df[~vol_df["is_ref"]]
                if not box_rows.empty:
                    box_vol_cm3 = float(box_rows.iloc[0]["real_vol_cm3"])
                if not obj_rows.empty:
                    obj_vol_cm3 = float(obj_rows.iloc[0]["real_vol_cm3"])

        total_time = time.time() - total_t0
        _print_summary(total_time, inference_time, ply_path, scene_recon_path,
                       recon_mesh_paths, scene_wt_path, wt_mesh_paths,
                       npz_path2, target_dir, target_images_dir)
    finally:
        if logger is not None:
            logger.stop_and_write(args.image_folder, args.output_dir, inference_time,
                                  obj_vol_cm3=obj_vol_cm3, box_vol_cm3=box_vol_cm3)
