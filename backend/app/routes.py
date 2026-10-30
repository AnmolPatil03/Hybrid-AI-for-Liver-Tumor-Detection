"""
API routes.

    GET  /                      health check
    POST /blood/extract         OCR a lab report
    POST /blood/assess          rule-based triage of confirmed markers
    POST /clinical/predict      XGBoost HCC risk
    GET  /clinical/tree/{i}     export one boosted tree
    POST /segment               segment a single CT slice
    POST /analyze               2.5D segmentation from three adjacent slices
    POST /report/generate       final report
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.schemas import (
    HealthResponse, SegmentationResponse, FullAnalysisResponse,
    BloodMarkers, PatientInfo, BloodExtractionResponse, BloodAssessmentResponse,
    ClinicalXGBFeatures, ClinicalRiskResponse, DecisionTreeResponse,
    ReportRequest, ReportResponse,
)
from app import blood_report
from app import report_generator
from src.segmentation_inference import segmentation_engine
from src.clinical_xgb_inference import clinical_xgb_engine


router = APIRouter()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the API status and whether models are loaded.",
)
async def health_check():
    """Health check endpoint to verify the API is running."""
    return HealthResponse(
        status="Liver Tumor Detection API is running",
        segmentation_model_loaded=segmentation_engine.is_loaded,
        clinical_model_loaded=clinical_xgb_engine.is_loaded,
    )


@router.post(
    "/blood/extract",
    response_model=BloodExtractionResponse,
    summary="Stage 1a — Extract Tumour Markers from a Blood Report",
    description=(
        "Upload a blood report (PDF or image). OCR extracts the tumour-marker "
        "values (AFP, AFP-L3%, DCP) and returns them alongside the raw OCR text "
        "so the user can review and correct them before assessment.\n\n"
        "The corrected values are then sent to `/blood/assess`."
    ),
)
async def blood_extract(file: UploadFile = File(..., description="Blood report (PDF/PNG/JPG)")):
    """OCR a blood report and parse marker values for user review."""
    allowed_types = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg']
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: PDF, PNG, JPG."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = blood_report.extract_from_upload(
            file_bytes, file.content_type, file.filename
        )
    except RuntimeError as e:
        # Tesseract not installed / OCR unavailable.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blood report extraction failed: {str(e)}")

    return BloodExtractionResponse(
        markers=BloodMarkers(**result["markers"]),
        patient=PatientInfo(**result["patient"]),
        clinical_features=ClinicalXGBFeatures(**result["clinical_features"]),
        raw_text=result["raw_text"],
    )


@router.post(
    "/blood/assess",
    response_model=BloodAssessmentResponse,
    summary="Stage 1b — Assess Confirmed Marker Values",
    description=(
        "Submit the user-confirmed tumour-marker values. A rule engine scores "
        "them against clinical thresholds and returns whether a CT scan should "
        "be uploaded next (`recommend_ct`). No ML model is involved in this "
        "deterministic, explainable step."
    ),
)
async def blood_assess(markers: BloodMarkers):
    """Score confirmed blood-marker values and decide if a CT scan is needed."""
    result = blood_report.assess_markers(markers.model_dump())
    return BloodAssessmentResponse(
        score=result["score"],
        risk_level=result["risk_level"],
        recommend_ct=result["recommend_ct"],
        flags=result["flags"],
        reasons=result["reasons"],
        interpretation=result["interpretation"],
    )


@router.post(
    "/clinical/predict",
    response_model=ClinicalRiskResponse,
    summary="Stage 1c — Clinical HCC Risk (standalone XGBoost)",
    description=(
        "Run the standalone XGBoost clinical-risk model over the confirmed "
        "clinical feature values (from /blood/extract's `clinical_features`, "
        "after user review/correction, or entered manually). No CT scan is "
        "involved here — clinical and imaging are independent stages.\n\n"
        "Decision rule: predicted risk probability > 0.5 -> `recommend_ct` is "
        "True (prompt a CT scan upload for imaging analysis); otherwise the "
        "result reports no significant risk of tumor.\n\n"
        "All fields are optional; anything missing is imputed by the model's "
        "training-fit preprocessor (not guessed at the OCR layer) and listed "
        "in `missing_features`."
    ),
)
async def clinical_predict(features: ClinicalXGBFeatures):
    """Run the standalone XGBoost clinical-risk model."""
    if not clinical_xgb_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Clinical XGBoost model not loaded. Train it first: python train_clinical_xgb.py",
        )

    try:
        result = clinical_xgb_engine.predict(features.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clinical risk prediction failed: {str(e)}")

    return ClinicalRiskResponse(**result)


@router.get(
    "/clinical/tree/{tree_index}",
    response_model=DecisionTreeResponse,
    summary="Export a Single XGBoost Decision Tree",
    description=(
        "Export one tree (index 0 to n_estimators-1) from the clinical XGBoost "
        "ensemble as nested JSON. `n_estimators` in the response gives the "
        "valid index range."
    ),
)
async def get_clinical_tree(tree_index: int):
    """Export a single tree from the clinical XGBoost ensemble for visualization."""
    if not clinical_xgb_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Clinical XGBoost model not loaded. Train it first: python train_clinical_xgb.py",
        )
    try:
        result = clinical_xgb_engine.get_tree_json(tree_index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return DecisionTreeResponse(**result)


@router.post(
    "/report/generate",
    response_model=ReportResponse,
    summary="Stage 3 — Generate the Final Report",
    description=(
        "Combine the already-computed results (patient demographics, blood "
        "assessment, clinical risk and CT analysis) into a structured, "
        "clinician-readable report. No new analysis is run at this step."
    ),
)
async def generate_report(req: ReportRequest):
    """Build the final report from the aggregated pipeline results."""
    try:
        # Report writing makes a blocking network call, so keep it off the event loop.
        result = await run_in_threadpool(report_generator.generate_report, req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    return ReportResponse(**result)


@router.post(
    "/segment",
    response_model=SegmentationResponse,
    summary="Segment Tumor — Get Location, Size & Grad-CAM",
    description=(
        "Upload a CT scan image and get the tumor segmentation mask, "
        "tumor location (centroid, bounding box), tumor size (area), and a "
        "Seg-Grad-CAM explainability overlay."
    ),
)
async def segment(file: UploadFile = File(..., description="CT scan image file")):
    """
    Segment the tumor in the uploaded CT scan.

    Returns tumor location (centroid, bounding box), size (pixel area,
    percentage), and a Grad-CAM explainability overlay.
    """
    allowed_types = ['image/png', 'image/jpeg', 'image/jpg']
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: PNG, JPG, JPEG."
        )

    if not segmentation_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model not loaded. Train it first: python train_segmentation_convnext.py"
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        result = segmentation_engine.predict_mask(image_bytes)
        return SegmentationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )


@router.post(
    "/analyze",
    response_model=FullAnalysisResponse,
    summary="Full CT Analysis — Segmentation + Grad-CAM",
    description=(
        "Upload the CT slice to analyze (`file`) plus its two immediate "
        "neighbouring slices (`prev_file`/`next_file`). The model is trained "
        "on 2.5D input, stacking [previous, center, next] as its 3 channels."
    ),
)
async def analyze(
    file: UploadFile = File(..., description="CT scan slice to analyze (center)"),
    prev_file: UploadFile = File(..., description="Previous adjacent CT slice (required)"),
    next_file: UploadFile = File(..., description="Next adjacent CT slice (required)"),
):
    """2.5D segmentation from the selected slice and its two neighbours, plus Grad-CAM."""
    allowed_types = ['image/png', 'image/jpeg', 'image/jpg']
    for f in (file, prev_file, next_file):
        if f.content_type and f.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {f.content_type}. Allowed: PNG, JPG, JPEG."
            )

    if not segmentation_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded.")

    try:
        image_bytes = await file.read()
        prev_bytes = await prev_file.read()
        next_bytes = await next_file.read()
        if not image_bytes or not prev_bytes or not next_bytes:
            raise HTTPException(
                status_code=400,
                detail="All three slices (file, prev_file, next_file) are required and must be non-empty."
            )

        result = segmentation_engine.predict_mask(image_bytes, prev_bytes=prev_bytes, next_bytes=next_bytes)
        return FullAnalysisResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
