"""Pydantic request/response models."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class TumorLocation(BaseModel):
    """Tumor centroid location."""
    centroid_x: float = Field(..., description="X coordinate of tumor center (pixels)")
    centroid_y: float = Field(..., description="Y coordinate of tumor center (pixels)")
    centroid_x_normalized: float = Field(..., description="Normalized X (0-1)")
    centroid_y_normalized: float = Field(..., description="Normalized Y (0-1)")


class TumorBBox(BaseModel):
    """Tumor bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float


class IntensityStats(BaseModel):
    """Grayscale pixel-intensity stats (0-255) within the tumor mask.

    These are raw image intensities, NOT calibrated Hounsfield Units — HU would
    require the DICOM rescale slope/intercept, which is not available here.
    """
    min: int = Field(..., description="Minimum grayscale value in the mask (0-255)")
    mean: float = Field(..., description="Mean grayscale value in the mask (0-255)")
    max: int = Field(..., description="Maximum grayscale value in the mask (0-255)")


class SegmentationResponse(BaseModel):
    """Response model for the /segment endpoint."""

    has_tumor: bool = Field(
        ...,
        description="Whether a tumor was detected in the segmentation mask"
    )
    tumor_area_pixels: int = Field(
        ...,
        description="Number of pixels classified as tumor"
    )
    tumor_area_percentage: float = Field(
        ...,
        description="Percentage of image area occupied by tumor"
    )
    tumor_location: Optional[TumorLocation] = Field(
        None,
        description="Centroid coordinates of the tumor"
    )
    tumor_bbox: Optional[TumorBBox] = Field(
        None,
        description="Bounding box around the tumor region"
    )
    tumor_intensity: Optional[IntensityStats] = Field(
        None,
        description="Grayscale intensity stats within the tumor mask (not calibrated HU)"
    )
    segmentation_overlay: Optional[str] = Field(
        None,
        description="Base64-encoded overlay image showing tumor region (PNG)"
    )
    segmentation_mask: Optional[str] = Field(
        None,
        description="Base64-encoded binary mask image (PNG)"
    )
    gradcam_image: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded Seg-Grad-CAM explainability overlay (PNG). Backprops from the "
            "aggregated segmentation output (no separate classifier exists in this system — "
            "see src/gradcam.py's SegGradCAM)."
        ),
    )


class FullAnalysisResponse(BaseModel):
    """Response model for the /analyze endpoint."""

    has_tumor: bool = Field(..., description="Tumor detected in segmentation")
    tumor_area_pixels: int = Field(default=0)
    tumor_area_percentage: float = Field(default=0.0)
    tumor_location: Optional[TumorLocation] = None
    tumor_bbox: Optional[TumorBBox] = None
    tumor_intensity: Optional[IntensityStats] = None
    segmentation_overlay: Optional[str] = None
    segmentation_mask: Optional[str] = None
    gradcam_image: Optional[str] = Field(
        None, description="Base64-encoded Seg-Grad-CAM explainability overlay (PNG)"
    )


class BloodMarkers(BaseModel):
    """
    Tumor markers (primary) and liver panel values (supporting) for the rule
    engine. Values OCR could not read are null.
    """
    afp: Optional[float] = Field(None, description="Alpha-Fetoprotein (ng/mL), ref ≤10")
    afp_l3: Optional[float] = Field(None, description="AFP-L3 as % of total AFP, ref <10")
    dcp: Optional[float] = Field(None, description="DCP / PIVKA-II (mAU/mL), ref <40")
    albumin: Optional[float] = Field(None, description="Albumin (g/dL), ref 3.5–5.0")
    platelets: Optional[float] = Field(None, description="Platelet count (×10⁹/L), ref 150–400")
    total_bilirubin: Optional[float] = Field(None, description="Total bilirubin (mg/dL), ref 0.1–1.2")


class PatientInfo(BaseModel):
    """Patient demographics parsed from the report."""
    name: Optional[str] = Field(None, description="Patient name as printed on the report")
    patient_id: Optional[str] = Field(None, description="Patient ID / UHID")
    age: Optional[int] = Field(None, description="Patient age in years")
    sex: Optional[str] = Field(None, description="'M' or 'F'")
    study_date: Optional[str] = Field(None, description="Collection / report date from the report")


class ClinicalXGBFeatures(BaseModel):
    """
    The 11 input features of the clinical XGBoost model. Null fields are
    imputed by the model's preprocessor. Returned by /blood/extract and
    accepted by /clinical/predict.
    """
    AFP_ng_mL: Optional[float] = Field(None, description="Alpha-Fetoprotein (ng/mL), ref 0.0-20.0")
    AFP_L3_percent: Optional[float] = Field(None, description="AFP-L3 as % of total AFP, ref <10.0")
    DCP_mAU_mL: Optional[float] = Field(None, description="DCP / PIVKA-II (mAU/mL), ref <40.0")
    Albumin_g_dL: Optional[float] = Field(None, description="Albumin (g/dL), ref 3.5-5.0")
    Bilirubin_mg_dL: Optional[float] = Field(None, description="Total bilirubin (mg/dL), ref 0.1-1.2")
    Platelet_count_x10e9_L: Optional[float] = Field(None, description="Platelet count (x10^9/L), ref 150-450")
    Age: Optional[float] = Field(None, description="Age in years")
    Sex: Optional[str] = Field(None, description="'M' or 'F'")
    HCV_status: Optional[int] = Field(None, description="1 = reactive/positive, 0 = non-reactive")
    HBV_status: Optional[int] = Field(None, description="1 = reactive/positive, 0 = non-reactive")
    Cirrhosis_status: Optional[int] = Field(None, description="1 = present, 0 = absent")


class DecisionTreeResponse(BaseModel):
    """One tree of the XGBoost ensemble as nested JSON, rendered by the frontend."""
    tree_index: int = Field(..., description="Index of the exported tree (0 to n_estimators-1)")
    n_estimators: int = Field(..., description="Total number of trees in the ensemble")
    tree: Dict[str, Any] = Field(
        ..., description="Nested tree structure from booster.get_dump(dump_format='json')"
    )


class ClinicalRiskResponse(BaseModel):
    """Response for /clinical/predict — the standalone XGBoost model's verdict."""
    risk_probability: float = Field(..., description="Predicted probability of HCC (0.0-1.0)", ge=0.0, le=1.0)
    risk_level: str = Field(..., description="'elevated' or 'normal'")
    recommend_ct: bool = Field(..., description="True if risk_probability > 0.5 — prompt CT scan upload")
    missing_features: List[str] = Field(
        default_factory=list, description="Feature names that were missing and imputed by the model layer"
    )
    interpretation: str = Field(..., description="Plain-language summary")


class BloodExtractionResponse(BaseModel):
    """Response for /blood/extract — OCR result for the user to review/correct."""
    markers: BloodMarkers = Field(..., description="Values parsed from the report")
    patient: PatientInfo = Field(
        default_factory=PatientInfo,
        description="Patient demographics parsed from the report",
    )
    clinical_features: ClinicalXGBFeatures = Field(
        default_factory=ClinicalXGBFeatures,
        description="Full 11-field clinical feature set parsed for the XGBoost model (/clinical/predict)",
    )
    raw_text: str = Field(..., description="Raw OCR text, for the user to verify against")


class BloodAssessmentResponse(BaseModel):
    """Response for /blood/assess — the rule-engine decision."""
    score: int = Field(..., description="Total risk score from the rule engine")
    risk_level: str = Field(..., description="'low', 'suspicious', or 'unknown'")
    recommend_ct: Optional[bool] = Field(
        None,
        description="True → prompt the user to upload a CT scan. None if no primary marker was provided."
    )
    flags: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-marker flag: 'normal' | 'borderline' | 'high' | 'low'"
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable reasons behind the score"
    )
    interpretation: str = Field(..., description="Plain-language summary for the report")


class ReportRequest(BaseModel):
    """
    Results collected by the frontend for /report/generate. All optional;
    base64 images are removed server-side.
    """
    patient: Optional[Dict[str, Any]] = Field(None, description="Patient demographics")
    clinical: Optional[Dict[str, Any]] = Field(None, description="Blood-marker values")
    assessment: Optional[Dict[str, Any]] = Field(None, description="Blood rule-engine assessment")
    analysis: Optional[Dict[str, Any]] = Field(None, description="CT analysis result")
    clinical_risk: Optional[Dict[str, Any]] = Field(None, description="Standalone XGBoost clinical-risk model result")


class ReportSection(BaseModel):
    """One titled section of the generated report."""
    title: str = Field(..., description="Section heading")
    body: str = Field(..., description="Section text ('\\n' newlines, '- ' bullets)")


class ReportResponse(BaseModel):
    """Response for /report/generate."""
    sections: List[ReportSection] = Field(..., description="Ordered report sections")
    summary: str = Field("", description="One-line impression")
    disclaimer: str = Field(..., description="Research/educational disclaimer")
    generated_by: str = Field(..., description="'gemini' or 'fallback' (template)")
    model: Optional[str] = Field(None, description="Gemini model id, or null for the template")


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = Field(
        ...,
        description="API status message",
        examples=["Liver Tumor Detection API is running"]
    )
    segmentation_model_loaded: bool = Field(
        default=False,
        description="Whether the segmentation model has been loaded"
    )
    clinical_model_loaded: bool = Field(
        default=False,
        description="Whether the clinical XGBoost model has been loaded"
    )
