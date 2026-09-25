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
from onnx_engine import UniversalRestorationEngine

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

# Base Paths
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONNX_TASK1_PATH = os.path.join(WORKSPACE_ROOT, "checkpoints", "universal_ae.onnx")
if not os.path.exists(ONNX_TASK1_PATH):
    ONNX_TASK1_PATH = os.path.join(WORKSPACE_ROOT, "universal-restoration", "checkpoints", "universal_ae.onnx")

# Find dataset images directory
POSSIBLE_IMG_DIRS = [
    os.path.join(WORKSPACE_ROOT, "images"),
    os.path.join(WORKSPACE_ROOT, "Images"),
]
IMAGES_DIR = next((d for d in POSSIBLE_IMG_DIRS if os.path.exists(d)), None)

# Initialize Task 1 Engine
task1_engine: Optional[UniversalRestorationEngine] = None
if os.path.exists(ONNX_TASK1_PATH):
    try:
        task1_engine = UniversalRestorationEngine(ONNX_TASK1_PATH)
    except Exception as e:
        print(f"Warning: Failed to load ONNX Task 1: {e}")


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
            "task_2_hard_routing": False,
            "task_3_soft_mixture": False,
            "task_4_face_to_sketch": False,
        },
        "onnx_model_loaded": task1_engine is not None,
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
# Task 2: Hard-Routing Endpoint (Placeholder for Task 2 Models)
# ==============================================================================

@app.post("/api/hard-routing")
async def hard_routing_restoration(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    mode: str = Form("predicted"),  # 'oracle' or 'predicted'
):
    """
    Task 2: Hard-Routed Restoration using Classifier + Specialist Autoencoders.
    Returns 4 classifier probabilities, selected expert, and restored image.
    """
    # Active placeholder until Task 2 models are trained
    return {
        "status": "Task 2 training in progress",
        "mode": mode,
        "probabilities": {
            "clean": 0.05,
            "salt_and_pepper": 0.88,
            "gaussian_blur": 0.04,
            "rectangular_occlusion": 0.03,
        },
        "predicted_corruption": "salt_and_pepper",
        "selected_expert": "Salt-and-Pepper Specialist Autoencoder",
        "latency_ms": 14.2,
        "message": "Task 2 classifier and specialists will be plugged in once trained.",
    }


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
