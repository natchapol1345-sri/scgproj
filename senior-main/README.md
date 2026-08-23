# VGGT Pipeline

Self-contained 3D reconstruction pipeline: images → point cloud → cleaned objects → watertight meshes → real-world volumes (ArUco-calibrated).

## Layout

```
senior/
├── run.py                       # entry point (executable)
├── viewer.py                    # PLY/STL viewer (executable)
├── requirements.txt
├── inputs
├── README.md
├── vggt/                        # VGGT model package (bundled)
├── pipeline/
│   ├── cli.py                   # argparse
│   ├── config.py                # constants (ArUco index, real size, model URL)
│   ├── orchestrator.py          # main() — runs Stages 1-7
│   ├── stages/
│   │   ├── inference.py         # Stage 1 — VGGT model inference
│   │   ├── pointcloud.py        # Stage 2 — filter + export PLY
│   │   ├── clean.py             # Stage 3 — level / outlier / floor / cluster
│   │   ├── reconstruct.py       # Stage 4 — Poisson reconstruction
│   │   ├── watertight.py        # Stage 5 — PyMeshFix watertight repair
│   │   ├── evaluate.py          # Stage 6 — multi-view screenshots
│   │   └── volume.py            # Stage 7 — ArUco-scaled real-world volumes
│   ├── core/
│   │   ├── plane.py             # RANSAC + leveling primitives
│   │   ├── cluster.py           # DBSCAN + ArUco-aware ranking
│   │   ├── fill.py              # bottom cap fill (alpha-shape hull)
│   │   ├── filters.py           # confidence + spatial outlier filters
│   │   └── mesh.py              # merge_meshes, verify_watertight, cleanup
│   └── utils/
│       └── seeding.py           # seed_everything (random / numpy / torch / o3d)
├── workers/
│   ├── recons_worker.py         # Poisson reconstruction subprocess
│   ├── recons_methods_worker.py # multi-method reconstruction subprocess
│   └── meshfix_worker.py        # PyMeshFix subprocess
└── tools/
    └── com_vol.py               # standalone mesh-vs-reference volume tool
```

## Install

```bash
# Recommended: virtual env or conda env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
./run.py                                          # uses ./baam/ as input
./run.py --image_folder ./baam/
./run.py --image_folder ./baam/ --output_dir output/
./run.py --image_folder ./baam/ --skip_mesh       # PLY only
./run.py --image_folder ./baam/ --evaluate        # multi-view screenshots
./run.py --image_folder ./vase/ --conf_thres 30
```

Or:
```bash
python run.py --image_folder ./baam/
```

### All flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-i`, `--image_folder` | str | `./inputs/baam/` | Input image folder |
| `--output_dir` | str | `./output/` | Output directory |
| `--conf_thres` | float | 45.0 | Confidence threshold percentile |
| `--prediction_mode` | choice | `pointmap` | `pointmap` or `depth` |
| `--mask_black_bg` | flag | off | Mask near-black background |
| `--mask_white_bg` | flag | off | Mask near-white background |
| `--skip_mesh` | flag | off | Skip clean+reconstruct (PLY only) |
| `--num_objects` | int | 2 | Objects to extract during cleaning |
| `--max_frames` | int | auto | Max frames (auto-set to 7 on MPS) |
| `--evaluate` | flag | off | Auto-capture screenshots |
| `--no-watertight` | flag | off | Skip watertight repair |
| `--no-fill` | flag | off | Skip bottom cap fill (auto-disabled with segmentation) |
| `--no-segment-leg` | flag | off | Disable leg surface segmentation |
| `--segment-height-axis` | choice | `z` | Height axis for leg cut (`x`/`y`/`z`) |
| `--recon-method` | choice | `poisson` | Default method for all objects (`poisson`, `ball_pivot`, `alpha_shape`, `poisson_omp1`) |
| `--box-recon-method` | choice | — | Override method for box/ArUco only |
| `--obj-recon-method` | choice | — | Override method for object/limb only |
| `--seed` | int | 42 | Random seed |
| `-l`, `--log`/`--no-log` | flag | on | Append metrics to log.csv |

Backends auto-detected: CUDA → MPS (Apple Silicon) → CPU.

## Object identification

After Stage 3, two objects are saved:
- `object_0.ply` = target (unknown volume)
- `object_1.ply` = ArUco reference cube (known: 14×14×14 cm)

Ranking by combined score: `cubeness × 0.6 + bw_ratio × 0.4`. Most cube-like + most black/white → ArUco.

Adjust constants in `pipeline/config.py`:
- `REFERENCE_OBJECT_INDEX` — which output index is the reference (default 1)
- `REFERENCE_REAL_SIZE_CM` — real edge length in cm (default 14.0)

## Outputs

Default `output/`:
```
output/
├── points.ply                   # filtered point cloud
├── predictions.npz              # VGGT raw predictions
├── clean_objects/
│   ├── object_0.ply
│   └── object_1.ply
├── mesh/
│   ├── object_0_recon.ply/.stl       # Poisson (object)
│   ├── object_0.ply/.stl             # watertight (object)
│   ├── object_1_recon.ply/.stl       # Poisson (ArUco)
│   ├── object_1.ply/.stl             # watertight (ArUco)
│   ├── scene_recon.ply/.stl
│   ├── scene_colour.ply/.stl
│   └── scene.ply/.stl
├── evaluation/                  # multi-view screenshots (--evaluate)
└── target/                      # demo_gradio-compatible layout
```

## View

```bash
./viewer.py output/points.ply
./viewer.py output/mesh/object_0.ply
```
