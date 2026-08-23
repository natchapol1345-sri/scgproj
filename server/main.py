"""
VGGT 3D Reconstruction Pipeline — FastAPI Server

Bridges the senior-main/ pipeline (PyTorch + VGGT-1B) with the web/ frontend.
Model is loaded once at startup and reused across jobs.
Only one job runs at a time (GPU memory constraint).
"""

import asyncio
import io
import os
import shutil
import sys
import time
import traceback
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup: make senior-main importable ────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SENIOR_MAIN = _PROJECT_ROOT / "senior-main"
_WEB_DIR = _PROJECT_ROOT / "web"

# Insert senior-main at front of sys.path so `pipeline.*` and `vggt.*` resolve
if str(_SENIOR_MAIN) not in sys.path:
    sys.path.insert(0, str(_SENIOR_MAIN))

# ── Pipeline imports (from senior-main/) ───────────────────────────────
import torch
from vggt.models.vggt import VGGT
from vggt.utils.device import get_device, aggressive_cleanup
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from pipeline.config import (
    IMAGE_EXTENSIONS,
    VGGT_MODEL_URL,
    DEFAULT_MAX_FRAMES_MPS,
    REFERENCE_REAL_SIZE_CM,
)
from pipeline.stages.pointcloud import export_ply
from pipeline.stages.clean import clean_and_extract
from pipeline.stages.reconstruct import reconstruct_mesh_stage
from pipeline.stages.watertight import watertight_stage
from pipeline.stages.volume import compute_volumes
from pipeline.utils.seeding import seed_everything

from vggt.utils.device import is_mps, autocast_on


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"


STAGE_NAMES = [
    "Model Inference",
    "PLY Export",
    "Clean & Extract",
    "Reconstruction",
    "Watertight Repair",
    "Evaluation",
    "Volume Computation",
]


@dataclass
class JobState:
    job_id: str
    image_folder: str
    output_dir: str
    ref_size_cm: float = 14.0
    stages: list[dict] = field(default_factory=list)
    error: str | None = None
    total_time: float | None = None
    inference_time: float | None = None
    result: dict | None = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.stages:
            self.stages = [
                {"id": i + 1, "name": STAGE_NAMES[i], "status": StageStatus.PENDING, "message": ""}
                for i in range(7)
            ]

    def set_stage(self, stage_id: int, status: StageStatus, message: str = ""):
        self.stages[stage_id - 1]["status"] = status
        self.stages[stage_id - 1]["message"] = message

    def to_status_dict(self) -> dict:
        overall = "pending"
        if any(s["status"] == StageStatus.RUNNING for s in self.stages):
            overall = "running"
        elif all(s["status"] in (StageStatus.DONE, StageStatus.SKIPPED) for s in self.stages):
            overall = "done"
        elif any(s["status"] == StageStatus.ERROR for s in self.stages):
            overall = "error"

        return {
            "job_id": self.job_id,
            "overall": overall,
            "stages": [
                {"id": s["id"], "name": s["name"], "status": s["status"], "message": s["message"]}
                for s in self.stages
            ],
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Global State
# ═══════════════════════════════════════════════════════════════════════

jobs: dict[str, JobState] = {}
_job_lock = asyncio.Lock()  # Only one pipeline job at a time
_model = None  # VGGT-1B model (loaded at startup)
_device = None  # torch device string


# ═══════════════════════════════════════════════════════════════════════
#  Model Loading (once at startup)
# ═══════════════════════════════════════════════════════════════════════

def load_vggt_model(device: str):
    """Load VGGT-1B model and move to device. Called once at startup."""
    print(f"[server] Loading VGGT-1B model on {device}...")
    t0 = time.time()
    model = VGGT()
    state_dict = torch.hub.load_state_dict_from_url(VGGT_MODEL_URL, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    elapsed = time.time() - t0
    print(f"[server] Model loaded in {elapsed:.1f}s")
    return model


# ═══════════════════════════════════════════════════════════════════════
#  Inference (uses pre-loaded model)
# ═══════════════════════════════════════════════════════════════════════

def run_inference_with_model(model, image_folder: str, device: str, max_frames: int | None = None):
    """Run VGGT inference using a pre-loaded model. Returns (predictions, inference_time)."""
    import glob

    aggressive_cleanup(device)

    if max_frames is None and is_mps(device):
        max_frames = DEFAULT_MAX_FRAMES_MPS

    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_names = [p for p in image_names if p.lower().endswith(IMAGE_EXTENSIONS)]
    if not image_names:
        raise ValueError(f"No images found in {image_folder}")

    # Subsample frames if needed
    n = len(image_names)
    if max_frames is not None and n > max_frames:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        indices = sorted(set(indices))
        image_names = [image_names[i] for i in indices]

    print(f"[server] Inference on {len(image_names)} images from {image_folder}")

    images = load_and_preprocess_images(image_names).to(device)

    t1 = time.time()
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
    print(f"[server] Inference done in {inference_time:.1f}s")

    aggressive_cleanup(device)
    return predictions, inference_time


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline Runner (blocking, runs in thread)
# ═══════════════════════════════════════════════════════════════════════

def _make_args(ref_size_cm: float = 14.0) -> SimpleNamespace:
    """Create an args-like namespace matching what pipeline stages expect."""
    return SimpleNamespace(
        prediction_mode="pointmap",
        conf_thres=45.0,
        mask_black_bg=False,
        mask_white_bg=False,
        skip_mesh=False,
        num_objects=2,
        max_frames=None,
        no_watertight=False,
        no_fill=False,
        segment_leg=True,
        segment_height_axis="z",
        recon_method="poisson",
        box_recon_method=None,
        obj_recon_method=None,
        voxel_res=150,
        auto_res=True,
        seed=42,
        log=False,
        ref_size_cm=ref_size_cm,
    )


def run_pipeline(job: JobState):
    """Run the full 7-stage pipeline. Called from a background thread."""
    global _model, _device

    args = _make_args(job.ref_size_cm)
    output_dir = job.output_dir
    image_folder = job.image_folder
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(args.seed)
    t0 = time.time()

    try:
        # ── Stage 1: Inference ──────────────────────────────────────
        job.set_stage(1, StageStatus.RUNNING, "Loading model and running inference...")
        predictions, inference_time = run_inference_with_model(
            _model, image_folder, _device, args.max_frames
        )
        job.inference_time = inference_time

        # Save predictions
        target_dir = os.path.join(output_dir, "target")
        target_images_dir = os.path.join(target_dir, "images")
        os.makedirs(target_images_dir, exist_ok=True)

        npz_path = os.path.join(output_dir, "predictions.npz")
        save_dict = {k: v for k, v in predictions.items() if v is not None}
        np.savez_compressed(npz_path, **save_dict)

        job.set_stage(1, StageStatus.DONE, f"Inference complete in {inference_time:.1f}s")

        # ── Stage 2: Export PLY ─────────────────────────────────────
        job.set_stage(2, StageStatus.RUNNING, "Exporting PLY point cloud...")
        ply_path = export_ply(predictions, output_dir, args)
        job.set_stage(2, StageStatus.DONE, f"Exported {ply_path}")

        # ── Stage 3: Clean & Extract ────────────────────────────────
        job.set_stage(3, StageStatus.RUNNING, "Cleaning point cloud and extracting objects...")
        object_paths = clean_and_extract(
            ply_path, output_dir, args.num_objects,
            seed=args.seed,
            segment_leg=args.segment_leg,
            segment_height_axis=args.segment_height_axis,
            fill_enabled=not args.no_fill,
        )
        if object_paths:
            job.set_stage(3, StageStatus.DONE, f"Extracted {len(object_paths)} objects")
        else:
            job.set_stage(3, StageStatus.ERROR, "No objects extracted")
            job.error = "Stage 3 failed: no objects extracted from point cloud"
            return

        # ── Stage 4: Reconstruction ─────────────────────────────────
        job.set_stage(4, StageStatus.RUNNING, "Reconstructing meshes (Poisson)...")
        scene_recon_path, recon_mesh_paths = reconstruct_mesh_stage(
            object_paths, output_dir,
            seed=args.seed,
            method=args.recon_method,
            box_method=args.box_recon_method,
            obj_method=args.obj_recon_method,
        )
        if recon_mesh_paths:
            job.set_stage(4, StageStatus.DONE, f"Reconstructed {len(recon_mesh_paths)} meshes")
        else:
            job.set_stage(4, StageStatus.ERROR, "Reconstruction failed")
            job.error = "Stage 4 failed: mesh reconstruction produced no output"
            return

        # ── Stage 5: Watertight ─────────────────────────────────────
        scene_wt_path = None
        wt_mesh_paths = []
        if recon_mesh_paths and not args.no_watertight:
            job.set_stage(5, StageStatus.RUNNING, "Making meshes watertight (PyMeshFix)...")
            scene_wt_path, wt_mesh_paths = watertight_stage(recon_mesh_paths, output_dir)
            if wt_mesh_paths:
                job.set_stage(5, StageStatus.DONE, f"Repaired {len(wt_mesh_paths)} meshes")
            else:
                job.set_stage(5, StageStatus.DONE, "Watertight repair skipped (no output)")
        else:
            job.set_stage(5, StageStatus.SKIPPED, "Watertight repair skipped")

        # ── Stage 6: Evaluation (skipped in web mode) ───────────────
        job.set_stage(6, StageStatus.SKIPPED, "Skipped — interactive 3D viewer available")

        # ── Stage 7: Volume Computation ─────────────────────────────
        vol_objects = wt_mesh_paths or recon_mesh_paths
        vol_df = None
        if vol_objects:
            job.set_stage(7, StageStatus.RUNNING, "Computing real-world volumes...")
            vol_df = compute_volumes(
                vol_objects,
                voxel_res=args.voxel_res,
                auto_res=args.auto_res,
            )
            if vol_df is not None:
                job.set_stage(7, StageStatus.DONE, "Volume computation complete")
            else:
                job.set_stage(7, StageStatus.DONE, "Volume computation: no reference found")
        else:
            job.set_stage(7, StageStatus.SKIPPED, "No meshes available for volume computation")

        total_time = time.time() - t0
        job.total_time = total_time

        # ── Build result dict ───────────────────────────────────────
        # Count points from PLY
        point_count = 0
        face_count = 0
        is_watertight = False
        try:
            import trimesh
            pc = trimesh.load(ply_path, process=False)
            point_count = len(pc.vertices) if hasattr(pc, 'vertices') else 0
        except Exception:
            pass

        # Count faces from the best mesh
        mesh_for_count = wt_mesh_paths[0] if wt_mesh_paths else (recon_mesh_paths[0] if recon_mesh_paths else None)
        if mesh_for_count:
            try:
                import trimesh
                m = trimesh.load(mesh_for_count, process=False)
                face_count = len(m.faces) if hasattr(m, 'faces') else 0
                is_watertight = getattr(m, 'is_watertight', False)
            except Exception:
                pass

        # Build volume table
        volume_rows = []
        k_value = None
        linear_scale_value = None
        if vol_df is not None:
            ref_rows = vol_df[vol_df["is_ref"]]
            if not ref_rows.empty:
                ref = ref_rows.iloc[0]
                if ref["volume"] > 0:
                    real_ref_vol = REFERENCE_REAL_SIZE_CM ** 3
                    k_value = real_ref_vol / ref["volume"]
                    linear_scale_value = k_value ** (1.0 / 3.0)

            for _, row in vol_df.iterrows():
                volume_rows.append({
                    "name": row["name"],
                    "is_ref": bool(row["is_ref"]),
                    "size_x_cm": round(float(row.get("size_x_cm", 0)), 2),
                    "size_y_cm": round(float(row.get("size_y_cm", 0)), 2),
                    "size_z_cm": round(float(row.get("size_z_cm", 0)), 2),
                    "real_vol_cm3": round(float(row.get("real_vol_cm3", 0)), 2),
                    "real_vol_L": round(float(row.get("real_vol_L", 0)), 3),
                    "method": row.get("method", "unknown"),
                })

        job.result = {
            "total_time": round(total_time, 1),
            "inference_time": round(inference_time, 1),
            "point_count": point_count,
            "face_count": face_count,
            "is_watertight": is_watertight,
            "volumes": volume_rows,
            "k": round(k_value, 2) if k_value else None,
            "linear_scale": round(linear_scale_value, 4) if linear_scale_value else None,
            "ref_size_cm": REFERENCE_REAL_SIZE_CM,
        }

        print(f"[server] Pipeline complete in {total_time:.1f}s")

    except Exception as e:
        job.error = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()
        print(f"[server] Pipeline error:\n{tb}")
        # Mark current running stage as error
        for s in job.stages:
            if s["status"] == StageStatus.RUNNING:
                s["status"] = StageStatus.ERROR
                s["message"] = str(e)
                break


# ═══════════════════════════════════════════════════════════════════════
#  FastAPI Application
# ═══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load VGGT model at startup."""
    global _model, _device
    _device = get_device()
    print(f"[server] Device: {_device}")
    _model = load_vggt_model(_device)
    yield
    # Cleanup
    del _model
    aggressive_cleanup(_device)


app = FastAPI(
    title="VGGT 3D Reconstruction Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ──────────────────────────────────────────────────────

@app.post("/api/jobs")
async def create_job(
    images: list[UploadFile] = File(...),
    ref_size_cm: float = Form(14.0),
):
    """Upload images and create a new pipeline job."""
    if not images:
        raise HTTPException(400, "No images uploaded")

    job_id = uuid.uuid4().hex[:12]

    # Save images to senior-main/inputs/<job_id>/
    input_dir = _SENIOR_MAIN / "inputs" / job_id
    input_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for img in images:
        ext = os.path.splitext(img.filename or "")[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        dest = input_dir / img.filename
        content = await img.read()
        dest.write_bytes(content)
        saved += 1

    if saved == 0:
        shutil.rmtree(input_dir, ignore_errors=True)
        raise HTTPException(400, "No valid image files uploaded")

    output_dir = str(_SENIOR_MAIN / "output" / job_id)

    job = JobState(
        job_id=job_id,
        image_folder=str(input_dir),
        output_dir=output_dir,
        ref_size_cm=ref_size_cm,
    )
    jobs[job_id] = job

    return {"job_id": job_id, "images_saved": saved}


@app.post("/api/jobs/{job_id}/run")
async def run_job(job_id: str):
    """Start the pipeline for a job (runs in background thread)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    # Check if any stage is already running
    if any(s["status"] == StageStatus.RUNNING for s in job.stages):
        raise HTTPException(409, "Job is already running")

    # Check if already completed
    if all(s["status"] in (StageStatus.DONE, StageStatus.SKIPPED) for s in job.stages):
        raise HTTPException(409, "Job already completed")

    # Try to acquire the lock (only one job at a time)
    if _job_lock.locked():
        raise HTTPException(429, "Another job is currently running. Please wait.")

    async def _run_in_background():
        async with _job_lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_pipeline, job)

    asyncio.create_task(_run_in_background())

    return {"status": "started", "job_id": job_id}


@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get current status of all pipeline stages."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job.to_status_dict()


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get pipeline results (volumes, timings, mesh stats)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if job.result is None:
        if job.error:
            raise HTTPException(500, f"Pipeline failed: {job.error}")
        raise HTTPException(202, "Pipeline still running")

    return job.result


@app.get("/api/jobs/{job_id}/files")
async def list_job_files(job_id: str):
    """List all output files with sizes."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    output_dir = Path(job.output_dir)
    if not output_dir.exists():
        return {"files": []}

    files = []
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(output_dir)
            files.append({
                "path": str(rel).replace("\\", "/"),
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            })

    return {"files": files}


@app.get("/api/jobs/{job_id}/files/{filepath:path}")
async def download_file(job_id: str, filepath: str):
    """Download a single output file."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    file_path = Path(job.output_dir) / filepath
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, f"File not found: {filepath}")

    # Security: ensure path is within output_dir
    try:
        file_path.resolve().relative_to(Path(job.output_dir).resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@app.get("/api/jobs/{job_id}/files.zip")
async def download_all_files(job_id: str):
    """Download all output files as a ZIP archive."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    output_dir = Path(job.output_dir)
    if not output_dir.exists():
        raise HTTPException(404, "No output files yet")

    # Create ZIP in memory
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                arcname = str(f.relative_to(output_dir))
                zf.write(f, arcname)

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=output_{job_id}.zip"},
    )


# ── Serve static web files ─────────────────────────────────────────────

# Mount web/ as static at root (must be last so API routes take priority)
app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="static")
