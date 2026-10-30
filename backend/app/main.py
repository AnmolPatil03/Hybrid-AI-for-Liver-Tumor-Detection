"""
FastAPI entry point. Models are loaded once at startup.

    uvicorn app.main:app --reload
"""

import os
import sys
from contextlib import asynccontextmanager

# Load backend/.env before anything reads os.environ.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.routes import router
from src.segmentation_inference import segmentation_engine
from src.clinical_xgb_inference import clinical_xgb_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    print("\n" + "=" * 60)
    print("  LIVER TUMOR DETECTION API — Starting Up")
    print("=" * 60 + "\n")

    seg_model_path = os.path.join('outputs', 'models', 'best_segmentation_convnext_model.pth')
    try:
        segmentation_engine.load_model(seg_model_path)
        print("[INFO] Segmentation model loaded successfully.")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] Segmentation (tumor size/location) will be unavailable.")
        print("[WARNING] Train it first: python train_segmentation_convnext.py")

    try:
        clinical_xgb_engine.load_model()
        print("[INFO] Clinical XGBoost model loaded successfully.")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] /clinical/predict will be unavailable.")
        print("[WARNING] Train it first: python train_clinical_xgb.py")

    yield

    print("\n[INFO] Shutting down Liver Tumor Detection API...")


app = FastAPI(
    title="Liver Tumor Detection API",
    description=(
        "Deep Learning-powered API for liver tumor segmentation from CT scan "
        "images. Uses a ConvNeXt-Tiny encoder + U-Net decoder, with a standalone "
        "XGBoost model for clinical (lab-marker) risk assessment. Provides "
        "Seg-Grad-CAM explainability visualizations for CT inputs.\n\n"
        "## Features\n"
        "- **Tumor Segmentation**: location, size, and mask from CT scans\n"
        "- **Clinical Risk Model**: XGBoost over lab markers, decision-tree inspectable\n"
        "- **Explainable AI**: Seg-Grad-CAM heatmap overlay on segmentation outputs\n\n"
        "## Usage\n"
        "Upload a CT scan image to the `/analyze` endpoint for segmentation + Grad-CAM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Browsers reject allow_origins=["*"] with credentials, so allow localhost on any port.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
