# VGGT 3D Reconstruction — Web Interface

Web-based interface for the VGGT (Visual Geometry Grounded Transformer) 3D reconstruction pipeline.  
Upload multi-view images → get watertight 3D meshes with real-world volume measurement.

## Architecture

```
project/
├── senior-main/          # VGGT pipeline (PyTorch, pipeline stages, workers)
│   ├── run.py            # CLI entry point (not used by web)
│   ├── pipeline/         # Stage implementations (imported by server)
│   ├── vggt/             # VGGT-1B model package
│   ├── workers/          # Subprocess workers (reconstruction, meshfix)
│   └── requirements.txt  # Pipeline Python dependencies
├── server/               # FastAPI backend
│   └── main.py           # API endpoints + pipeline orchestration
├── web/                  # Frontend (HTML/CSS/JS + Three.js)
│   ├── index.html
│   ├── style.css
│   └── app.js            # API integration + Three.js 3D viewer
└── requirements-server.txt  # Server Python dependencies (FastAPI, uvicorn)
```

## Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (recommended) — VGGT-1B requires ~4–6 GB VRAM
  - Also works on Apple Silicon (MPS) or CPU (much slower)
- **~2 GB disk space** for VGGT-1B model weights (auto-downloaded on first run)

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install pipeline dependencies

```bash
pip install -r senior-main/requirements.txt
```

> **Note:** PyTorch with CUDA support should be installed first if you have a GPU.  
> See [pytorch.org](https://pytorch.org/get-started/locally/) for platform-specific instructions.

### 3. Install server dependencies

```bash
pip install -r requirements-server.txt
```

### 4. VGGT-1B Model Weights

The model weights (~2 GB) are **automatically downloaded** on first server startup via `torch.hub`.  
They are cached in `~/.cache/torch/hub/` and reused on subsequent runs.

To pre-download manually:
```python
import torch
torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
    map_location="cpu"
)
```

## Running

### Start the server

```bash
# From the project root directory:
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

On first startup, the server will:
1. Auto-detect the best device (CUDA → MPS → CPU)
2. Load the VGGT-1B model into memory (~30–60 seconds)
3. Start serving the web UI and API

### Open the web interface

Navigate to: **http://localhost:8000**

## Usage

1. **Input** — Upload 3–7 multi-view images of your object (with an ArUco reference cube visible)
2. **Run Pipeline** — Click to start the 7-stage reconstruction pipeline
3. **Processing** — Watch real-time progress as each stage completes
4. **Results** — Explore the 3D mesh in the interactive viewer, view volume measurements, and download output files

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/jobs` | Upload images + create job |
| `POST` | `/api/jobs/{id}/run` | Start pipeline execution |
| `GET` | `/api/jobs/{id}/status` | Poll stage-by-stage progress |
| `GET` | `/api/jobs/{id}/result` | Get results (volumes, timings) |
| `GET` | `/api/jobs/{id}/files` | List output files |
| `GET` | `/api/jobs/{id}/files/{path}` | Download single file |
| `GET` | `/api/jobs/{id}/files.zip` | Download all outputs as ZIP |

## Pipeline Stages

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Inference** | VGGT-1B forward pass → depth maps, point maps, camera poses |
| 2 | **PLY Export** | Confidence filter + outlier removal → colored point cloud |
| 3 | **Clean & Extract** | RANSAC leveling, floor removal, DBSCAN clustering → object + ArUco separation |
| 4 | **Reconstruction** | Poisson surface reconstruction per object |
| 5 | **Watertight** | PyMeshFix hole-filling + color transfer |
| 6 | **Evaluation** | (Skipped in web mode — interactive 3D viewer replaces screenshots) |
| 7 | **Volume** | ArUco-calibrated real-world volume computation |

## Troubleshooting

- **Out of Memory**: VGGT requires significant GPU memory. Reduce the number of input images (3–5 recommended) or use `--max_frames` flag
- **Slow inference**: On CPU, inference can take 10+ minutes. Use CUDA for ~8–15 second inference
- **Port in use**: Change the port with `--port 8001`
