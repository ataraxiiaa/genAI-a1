"""
FastAPI Backend for Generative AI Image Restoration & Synthesis Systems
Serves all 4 Tasks:
  1. Universal Restoration (Active with ONNX)
  2. Hard-Routed Restoration (Task 2)
  3. Soft Mixture-of-Experts Restoration (Task 3)
  4. Face-to-Sketch Generator (Task 4)
"""

import io
import os
import random
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from corruptions import corrupt_image, TEST_SEVERITY_CONFIGS
from onnx_engine import UniversalRestorationEngine, HardRoutingEngine

app = FastAPI(
    title="Generative AI Multi-Task Restoration & Synthesis API",
    description="Backend API serving ONNX models for Tasks 1, 2, 3, and 4",
    version="1.0.0",
)

# Enable CORS for local development (React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base Paths (Supports both local host execution and Docker container /app paths)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

POSSIBLE_CHECKPOINT_DIRS = [
    os.path.join(WORKSPACE_ROOT, "checkpoints"),
    os.path.join(CURRENT_DIR, "checkpoints"),
    os.path.join(WORKSPACE_ROOT, "universal-restoration", "checkpoints"),
    os.path.join(CURRENT_DIR, "universal-restoration", "checkpoints"),
    os.path.join(WORKSPACE_ROOT, "specialist-autoencoders", "checkpoints"),
    os.path.join(CURRENT_DIR, "specialist-autoencoders", "checkpoints"),
]

POSSIBLE_ONNX_PATHS = [
    os.path.join(cdir, "universal_ae.onnx") for cdir in POSSIBLE_CHECKPOINT_DIRS
]
ONNX_TASK1_PATH = next((p for p in POSSIBLE_ONNX_PATHS if os.path.exists(p)), POSSIBLE_ONNX_PATHS[0])

# Find dataset images directory
POSSIBLE_IMG_DIRS = [
    os.path.join(WORKSPACE_ROOT, "images"),
    os.path.join(CURRENT_DIR, "images"),
    os.path.join(WORKSPACE_ROOT, "Images"),
    os.path.join(CURRENT_DIR, "Images"),
]
IMAGES_DIR = next((d for d in POSSIBLE_IMG_DIRS if os.path.exists(d)), None)

# Initialize Task 1 Engine
task1_engine: Optional[UniversalRestorationEngine] = None
if os.path.exists(ONNX_TASK1_PATH):
    try:
        task1_engine = UniversalRestorationEngine(ONNX_TASK1_PATH)
        print(f"Loaded Task 1 ONNX model from: {ONNX_TASK1_PATH}")
    except Exception as e:
        print(f"Warning: Failed to load ONNX Task 1: {e}")
else:
    print(f"Warning: Task 1 ONNX model not found at candidate paths: {POSSIBLE_ONNX_PATHS}")

# Initialize Task 2 Engine (Hard-Routing Engine)
task2_engine: Optional[HardRoutingEngine] = None
try:
    task2_engine = HardRoutingEngine(POSSIBLE_CHECKPOINT_DIRS)
    if task2_engine.is_ready():
        print("Loaded Task 2 Hard-Routing ONNX Engine (Classifier + 3 Specialists)")
    else:
        print("Notice: Task 2 Hard-Routing engine initialized in partial/fallback mode.")
except Exception as e:
    print(f"Warning: Failed to initialize Task 2 engine: {e}")


# ==============================================================================
# General & Helper Endpoints
# ==============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint required by assignment."""
    return {
        "status": "healthy",
        "tasks": {
            "task_1_universal": task1_engine is not None,
            "task_2_hard_routing": task2_engine is not None and task2_engine.is_ready(),
            "task_3_soft_mixture": False,
            "task_4_face_to_sketch": False,
        },
        "onnx_model_loaded": task1_engine is not None,
        "specialists_loaded": task2_engine is not None and task2_engine.is_ready(),
    }



@app.get("/api/sample-images")
def get_sample_images():
    """Returns a list of sample pet images available in the dataset for quick testing."""
    if not IMAGES_DIR or not os.path.exists(IMAGES_DIR):
        return {"samples": []}

    all_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    # Return 16 diverse samples
    random.seed(42)
    sample_names = random.sample(all_files, min(16, len(all_files)))
    return {"samples": sample_names}


@app.get("/api/sample-images/{filename}")
def get_sample_image_data(filename: str):
    """Returns sample image as base64 for preview."""
    if not IMAGES_DIR:
        raise HTTPException(status_code=404, detail="Images directory not found")

    file_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample image not found")

    pil_img = Image.open(file_path).convert("RGB")
    if task1_engine:
        b64 = task1_engine.to_base64(pil_img.resize((128, 128)))
    else:
        buf = io.BytesIO()
        pil_img.resize((128, 128)).save(buf, format="JPEG")
        import base64
        b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    return {"filename": filename, "image_base64": b64}


# ==============================================================================
# Task 1: Universal Restoration Endpoint
# ==============================================================================

@app.post("/api/universal-restoration")
async def universal_restoration(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    corruption: str = Form("none"),
    severity: str = Form("medium"),
):
    """
    Task 1: Universal Multi-Corruption Denoising Autoencoder.
    Accepts uploaded image or sample name, applies optional corruption, runs ONNX model,
    and returns input, restored, error map, metrics, and inference latency.
    """
    if task1_engine is None:
        raise HTTPException(status_code=503, detail="Task 1 ONNX model not loaded on server.")

    # 1. Load source image
    if file and file.filename:
        content = await file.read()
        pil_clean = Image.open(io.BytesIO(content)).convert("RGB")
    elif sample_name and IMAGES_DIR:
        sample_path = os.path.join(IMAGES_DIR, sample_name)
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail=f"Sample {sample_name} not found")
        pil_clean = Image.open(sample_path).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or a valid sample_name.")

    # 2. Apply corruption
    if corruption == "none":
        # Image is already corrupted (or tested as clean)
        pil_corrupted = pil_clean.resize((128, 128), Image.Resampling.BILINEAR)
        corrupted_meta = {"type": "none"}
        ground_truth_for_error_map = None
    else:
        pil_corrupted, corrupted_meta = corrupt_image(pil_clean, corruption, severity)
        ground_truth_for_error_map = pil_clean.resize((128, 128), Image.Resampling.BILINEAR)

    # 3. Run ONNX inference
    result = task1_engine.run_restoration(
        corrupted_pil=pil_corrupted,
        clean_pil=ground_truth_for_error_map,
    )

    result["corruption_applied"] = corrupted_meta
    result["model_used"] = "Universal Denoising Autoencoder (ONNX)"
    return result


# ==============================================================================
# Task 2: Hard-Routing Endpoint (Live ONNX Classifier + Specialists)
# ==============================================================================

@app.post("/api/hard-routing")
async def hard_routing_restoration(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    corruption: str = Form("none"),
    severity: str = Form("medium"),
    mode: str = Form("predicted"),  # 'oracle' or 'predicted'
    oracle_label: Optional[int] = Form(None),
):
    """
    Task 2: Hard-Routed Restoration using Classifier + Specialist Autoencoders.
    Dispatches input image based on corruption prediction (or oracle ground truth),
    and returns 4 classifier probabilities, selected expert, restored image, and latency.
    """
    if task2_engine is None or not task2_engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Task 2 Hard-Routing engine is not fully loaded. Verify that specialist ONNX models exist.",
        )

    # 1. Load clean ground-truth image
    pil_clean = None
    if file and file.filename:
        contents = await file.read()
        pil_clean = Image.open(io.BytesIO(contents)).convert("RGB")
    elif sample_name and IMAGES_DIR:
        file_path = os.path.join(IMAGES_DIR, sample_name)
        if os.path.exists(file_path):
            pil_clean = Image.open(file_path).convert("RGB")

    if pil_clean is None:
        raise HTTPException(status_code=400, detail="Either an uploaded image file or a valid sample_name must be provided.")

    # 2. Apply requested corruption if any
    if corruption == "none":
        pil_corrupted = pil_clean
        corrupted_meta = {"type": "clean", "severity": "none"}
        ground_truth_for_error_map = None
    else:
        pil_corrupted, corrupted_meta = corrupt_image(pil_clean, corruption, severity)
        ground_truth_for_error_map = pil_clean.resize((128, 128), Image.Resampling.BILINEAR)

    # Infer oracle label if mode is oracle and label not explicitly provided
    if mode == "oracle" and oracle_label is None:
        c_norm = corruption.lower()
        if "salt" in c_norm or "noise" in c_norm:
            oracle_label = 1
        elif "blur" in c_norm:
            oracle_label = 2
        elif "occlusion" in c_norm or "box" in c_norm or "rect" in c_norm:
            oracle_label = 3
        else:
            oracle_label = 0

    # 3. Run Hard-Routing Engine
    result = task2_engine.run_hard_routing(
        corrupted_pil=pil_corrupted,
        clean_pil=ground_truth_for_error_map,
        mode=mode,
        oracle_label=oracle_label,
    )

    result["corruption_applied"] = corrupted_meta
    result["model_used"] = f"Task 2 Hard-Routing ({result['selected_expert']})"
    return result



# ==============================================================================
# Task 3: Soft Mixture-of-Experts Endpoint (Placeholder for Task 3 Models)
# ==============================================================================

@app.post("/api/soft-mixture")
async def soft_mixture_restoration(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    temperature: float = Form(1.0),
):
    """
    Task 3: Differentiable Soft Mixture-of-Experts.
    Returns 4 continuous gating weights [w_clean, w_salt, w_blur, w_occlusion] and blended output.
    """
    # Active placeholder until Task 3 is trained
    return {
        "status": "Task 3 training in progress",
        "temperature": temperature,
        "routing_weights": {
            "clean_identity": 0.08,
            "salt_and_pepper_expert": 0.72,
            "gaussian_blur_expert": 0.12,
            "occlusion_expert": 0.08,
        },
        "dominant_expert": "salt_and_pepper_expert",
        "latency_ms": 18.5,
        "message": "Task 3 Soft MoE gating network will be plugged in once trained.",
    }


# ==============================================================================
# Task 4: Face-to-Sketch Endpoint (Placeholder for Task 4 GAN)
# ==============================================================================

@app.post("/api/face-to-sketch")
async def face_to_sketch_generation(
    file: Optional[UploadFile] = File(None),
    style: str = Form("Style 1"),
):
    """
    Task 4: Style-Conditioned Face-to-Sketch Conditional GAN.
    Accepts facial photograph, generates corresponding sketch in Style 1, 2, or 3.
    """
    # Active placeholder until Task 4 cGAN is trained
    return {
        "status": "Task 4 training in progress",
        "selected_style": style,
        "supported_styles": ["Style 1", "Style 2", "Style 3"],
        "message": "Task 4 FS2K Conditional GAN generator will be plugged in once trained.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
