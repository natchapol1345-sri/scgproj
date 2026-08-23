"""Command-line argument parser for the pipeline."""
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT — run full pipeline: inference → PLY → clean → mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                  # default: ./baam/ input
  python run.py --image_folder examples/kitchen/images/ --conf_thres 30
  python run.py --skip_mesh                      # PLY only
""",
    )
    p.add_argument("-i", "--image_folder", type=str, default="./inputs/baam/",
                   help="Path to folder containing input images (default: ./baam/)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for output files (default: ./output/)")
    p.add_argument("--conf_thres", type=float, default=45.0,
                   help="Confidence threshold (percentile): filter bottom N%% of points (default: 45)")
    p.add_argument("--prediction_mode", type=str, default="pointmap",
                   choices=["pointmap", "depth"],
                   help="'pointmap' uses direct 3D point regression; 'depth' unprojects depth maps")
    p.add_argument("--mask_black_bg", action="store_true",
                   help="Mask out near-black background pixels")
    p.add_argument("--mask_white_bg", action="store_true",
                   help="Mask out near-white background pixels")
    p.add_argument("--skip_mesh", action="store_true",
                   help="Skip clean + reconstruct stages (PLY export only)")
    p.add_argument("--num_objects", type=int, default=2,
                   help="Number of objects to extract during cleaning (default: 2)")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Max frames to process (auto-set to 7 on MPS to avoid OOM)")
    p.add_argument("--no-watertight", action="store_true",
                   help="Skip watertight repair (export only Poisson reconstruction)")
    p.add_argument("--no-fill", action="store_true",
                   help="Skip bottom cap fill during cleaning")
    p.add_argument("--no-segment-leg", action="store_false", dest="segment_leg",
                   help="Disable marker-based leg surface segmentation (enabled by default)")
    p.add_argument("--segment-height-axis", type=str, default="z", choices=["x", "y", "z"],
                   help="Height axis for leg cut (default: z, vertical after leveling)")
    p.add_argument("--recon-method", type=str, default="poisson",
                   choices=["poisson", "ball_pivot", "alpha_shape", "poisson_omp1"],
                   help="Reconstruction method for all objects (default: poisson)")
    p.add_argument("--box-recon-method", type=str, default=None,
                   choices=["poisson", "ball_pivot", "alpha_shape", "poisson_omp1"],
                   help="Override recon method for box (ArUco) only")
    p.add_argument("--obj-recon-method", type=str, default=None,
                   choices=["poisson", "ball_pivot", "alpha_shape", "poisson_omp1"],
                   help="Override recon method for object (limb) only")
    p.add_argument("--voxel-res", type=int, default=150, dest="voxel_res",
                   help="Voxel grid resolution for volume (default: 150; ignored when --auto-res)")
    p.add_argument("--no-auto-res", action="store_false", dest="auto_res",
                   help="Fix voxel resolution instead of auto-tuning until convergence")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("-l", "--log", action=argparse.BooleanOptionalAction, default=True,
                   help="Append per-run metrics row to log.csv (default: on; use --no-log to disable)")
    return p.parse_args()
