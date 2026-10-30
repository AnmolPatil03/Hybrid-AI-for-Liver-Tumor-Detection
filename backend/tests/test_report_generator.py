"""
Tests for app.report_generator: image stripping and the template fallback
(no Gemini key needed).

    python -m pytest tests/ -q
"""

from app import report_generator as rg


SAMPLE_REQ = {
    "patient": {"name": "Mr. Anil Mehta", "patient_id": "PT-2026-04902", "age": 54, "sex": "M", "study_date": "02-Jul-2026"},
    "clinical": {"afp": 620, "dcp": 340, "albumin": 3.1, "platelets": 98},
    "assessment": {"score": 6, "risk_level": "suspicious", "flags": {"afp": "high", "dcp": "high", "albumin": "low"}, "reasons": ["AFP markedly elevated (620 ng/mL)"]},
    "analysis": {
        "prediction": "Tumor", "confidence": 0.94, "has_tumor": True,
        "tumor_area_pixels": 3847, "tumor_area_percentage": 5.87,
        "tumor_location": {"centroid_x": 156, "centroid_y": 132},
        "gradcam_image": "BIGBLOB", "segmentation_overlay": "BIGBLOB", "segmentation_mask": "BIGBLOB",
    },
    "clinical_risk": {"risk_probability": 0.81, "risk_level": "elevated", "recommend_ct": True,
                      "missing_features": [], "interpretation": "Clinical risk model estimates an 81% probability of HCC."},
}


def test_bundle_strips_base64_images():
    bundle = rg.build_findings_bundle(SAMPLE_REQ)
    img = bundle["imaging_analysis"]
    for k in ("gradcam_image", "segmentation_overlay", "segmentation_mask"):
        assert k not in img, f"{k} should be stripped from the bundle"
    # Non-image imaging fields are preserved.
    assert img["prediction"] == "Tumor"
    assert img["tumor_area_pixels"] == 3847


def test_bundle_handles_missing_sections():
    bundle = rg.build_findings_bundle({})
    assert bundle["patient"] == {}
    assert bundle["clinical_markers"] == {}
    assert bundle["imaging_analysis"] == {}
    # Must not raise and must not mutate anything into None.
    assert isinstance(bundle["clinical_risk_model"], dict)


def test_bundle_does_not_mutate_caller_analysis():
    req = {"analysis": {"prediction": "Tumor", "gradcam_image": "BLOB"}}
    rg.build_findings_bundle(req)
    # The original dict passed in must still contain its image blob.
    assert req["analysis"]["gradcam_image"] == "BLOB"


def test_fallback_report_shape_and_content():
    bundle = rg.build_findings_bundle(SAMPLE_REQ)
    out = rg._fallback_report(bundle)
    titles = [s["title"] for s in out["sections"]]
    assert titles == [
        "Patient Summary", "Clinical (Blood) Findings", "Clinical Risk Model",
        "Imaging (CT) Findings", "Impression", "Recommendations",
    ]
    joined = "\n".join(s["body"] for s in out["sections"])
    assert "Mr. Anil Mehta" in joined
    assert "PT-2026-04902" in joined
    assert "620" in joined            # AFP value
    assert "Tumor" in out["summary"]


def test_generate_report_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = rg.generate_report(SAMPLE_REQ)
    assert out["generated_by"] == "fallback"
    assert out["model"] is None
    assert out["disclaimer"]
    assert len(out["sections"]) >= 4
