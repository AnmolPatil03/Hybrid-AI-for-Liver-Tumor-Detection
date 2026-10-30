"""
Final report generation.

Builds a structured report from the results already produced by the
pipeline (rule-engine assessment, XGBoost risk, CT segmentation): Patient
Summary, Clinical Findings, Clinical Risk Model, Imaging Findings,
Impression, Recommendations.

* The report text is written through the Gemini API when GEMINI_API_KEY is
  set, restricted to the values provided.
* Otherwise, or if the call fails, the report is built from a template.
* The API key is read from backend/.env and never sent to the frontend.
"""

import json
import os
from typing import Any, Optional

# Load backend/.env (GEMINI_API_KEY, GEMINI_MODEL) if python-dotenv is present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


# Override with the exact model id from Google's current docs if needed.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

DISCLAIMER = (
    "Decision-support summary of the system's model outputs. "
    "For research and educational use only — not a clinical diagnosis. "
    "All findings must be confirmed by a qualified clinician."
)

# Base64 images are dropped before the results are sent for report writing.
_IMAGE_KEYS = ("gradcam_image", "segmentation_overlay", "segmentation_mask")

# Marker key -> (label, unit) for the deterministic fallback report.
_MARKER_META = {
    "afp": ("Alpha-Fetoprotein (AFP)", "ng/mL"),
    "afp_l3": ("AFP-L3", "%"),
    "dcp": ("DCP / PIVKA-II", "mAU/mL"),
    "albumin": ("Albumin", "g/dL"),
    "platelets": ("Platelet Count", "×10⁹/L"),
    "total_bilirubin": ("Total Bilirubin", "mg/dL"),
}


# --------------------------------------------------------------------------- #
# Bundle building
# --------------------------------------------------------------------------- #

def build_findings_bundle(req: dict) -> dict:
    """
    Normalise the incoming request into a clean, image-free findings bundle that
    is safe to serialise into a prompt. Missing sections simply stay empty.
    """
    patient = req.get("patient") or {}
    assessment = req.get("assessment") or {}
    clinical = req.get("clinical") or {}
    analysis = dict(req.get("analysis") or {})
    clinical_risk = req.get("clinical_risk") or {}

    for k in _IMAGE_KEYS:
        analysis.pop(k, None)

    return {
        "patient": patient,
        "clinical_markers": clinical,
        "blood_assessment": assessment,
        "imaging_analysis": analysis,
        "clinical_risk_model": clinical_risk,
    }


# --------------------------------------------------------------------------- #
# Gemini call
# --------------------------------------------------------------------------- #

_SYSTEM_INSTRUCTION = (
    "You are a clinical report writer for a liver-tumour decision-support tool. "
    "You are given the structured outputs of validated models (a blood-marker "
    "rule engine, a standalone XGBoost clinical-risk model, and a CT "
    "segmentation model — these are independent stages, not a "
    "fused/combined model). Write a professional, concise report that a "
    "hepatologist or radiologist could read.\n\n"
    "Strict rules:\n"
    "- Use ONLY the values provided. Never invent or estimate numbers.\n"
    "- If a value is missing, say 'not available' — do not guess.\n"
    "- Do NOT offer a second diagnostic opinion; report the models' findings.\n"
    "- Be factual and neutral; avoid alarming language beyond what the data supports.\n"
    "- Always frame the output as research/educational decision support, not a "
    "clinical diagnosis.\n\n"
    "Return ONLY valid JSON of the form:\n"
    '{"summary": "<one-sentence impression>", "sections": ['
    '{"title": "<section title>", "body": "<section text; use \\n newlines and '
    '\'- \' prefixed lines for bullet points>"}]}\n'
    "Include these sections in order when the data allows: Patient Summary, "
    "Clinical (Blood) Findings, Clinical Risk Model, Imaging (CT) Findings, "
    "Impression, Recommendations."
)


def _render_user_prompt(bundle: dict) -> str:
    return (
        "Generate the final liver-tumour report from the following structured "
        "model outputs (JSON):\n\n"
        + json.dumps(bundle, indent=2, ensure_ascii=False)
    )


def _call_gemini(bundle: dict) -> dict:
    """Call Gemini and return {'summary', 'sections'}. Raises on any failure."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    # Imported lazily so the template fallback works without the SDK.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=_render_user_prompt(bundle),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    raw = (resp.text or "").strip()
    data = json.loads(raw)
    sections = [
        {"title": str(s["title"]), "body": str(s["body"])}
        for s in data.get("sections", [])
        if isinstance(s, dict) and s.get("title") and s.get("body")
    ]
    if not sections:
        # Unexpected shape: use the raw text as a single section.
        sections = [{"title": "Radiology Report", "body": raw}]
    return {"summary": str(data.get("summary", "")), "sections": sections}


# --------------------------------------------------------------------------- #
# Deterministic fallback report
# --------------------------------------------------------------------------- #

def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "not available"
    if isinstance(v, float):
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return str(v)


def _fmt_pct(prob: Any) -> str:
    return "not available" if prob is None else f"{round(float(prob) * 100)}%"


def _fallback_report(bundle: dict) -> dict:
    """Build a deterministic, structured report directly from the data."""
    p = bundle.get("patient") or {}
    markers = bundle.get("clinical_markers") or {}
    blood = bundle.get("blood_assessment") or {}
    img = bundle.get("imaging_analysis") or {}
    risk = bundle.get("clinical_risk_model") or {}
    sections = []

    # Patient Summary
    sex = p.get("sex")
    age = p.get("age")
    age_sex = f"{age}/{sex}" if age is not None and sex else (str(age) if age is not None else "not available")
    sections.append({
        "title": "Patient Summary",
        "body": (
            f"- Name: {p.get('name') or 'not available'}\n"
            f"- Patient ID: {p.get('patient_id') or 'not available'}\n"
            f"- Age/Sex: {age_sex}\n"
            f"- Study date: {p.get('study_date') or 'not available'}"
        ),
    })

    # Clinical (Blood) Findings
    flags = blood.get("flags") or {}
    marker_lines = []
    for key, (label, unit) in _MARKER_META.items():
        if key in markers and markers[key] is not None:
            flag = flags.get(key)
            tag = f" [{flag}]" if flag and flag != "normal" else ""
            marker_lines.append(f"- {label}: {_fmt_num(markers[key])} {unit}{tag}")
    blood_body = "\n".join(marker_lines) if marker_lines else "- No blood-marker values provided."
    if blood:
        blood_body += (
            f"\n- Blood risk level: {blood.get('risk_level', 'not available')} "
            f"(score {blood.get('score', 'n/a')})"
        )
        for reason in (blood.get("reasons") or []):
            blood_body += f"\n- {reason}"
    sections.append({"title": "Clinical (Blood) Findings", "body": blood_body})

    # Clinical Risk Model
    if risk:
        missing = risk.get("missing_features") or []
        sections.append({
            "title": "Clinical Risk Model",
            "body": (
                f"- Predicted HCC risk probability: {_fmt_pct(risk.get('risk_probability'))}\n"
                f"- Risk level: {risk.get('risk_level', 'not available')}\n"
                f"- CT scan recommended: {'yes' if risk.get('recommend_ct') else 'no'}"
                + (f"\n- Imputed (missing) features: {', '.join(missing)}" if missing else "")
            ),
        })

    # Imaging (CT) Findings
    if img:
        loc = img.get("tumor_location") or {}
        loc_str = (
            f"({_fmt_num(loc.get('centroid_x'), 0)}, {_fmt_num(loc.get('centroid_y'), 0)}) px"
            if loc else "not available"
        )
        sections.append({
            "title": "Imaging (CT) Findings",
            "body": (
                f"- Tumour detected on segmentation: {'yes' if img.get('has_tumor') else 'no'}\n"
                f"- Tumour area: {_fmt_num(img.get('tumor_area_percentage'))}% of image "
                f"({_fmt_num(img.get('tumor_area_pixels'), 0)} px)\n"
                f"- Tumour centroid: {loc_str}"
            ),
        })
    else:
        sections.append({
            "title": "Imaging (CT) Findings",
            "body": "- No CT imaging analysis was provided.",
        })

    # Impression: use the imaging finding when available, otherwise the clinical risk.
    if img:
        final = "Tumor" if img.get("has_tumor") else "Normal"
    elif risk:
        final = "Tumor" if risk.get("recommend_ct") else "Normal"
    else:
        final = "Indeterminate"
    impression = (
        risk.get("interpretation")
        or blood.get("interpretation")
        or f"Overall assessment: {final}."
    )
    sections.append({"title": "Impression", "body": impression})

    # Recommendations
    if str(final).lower() == "tumor":
        rec = (
            "- Correlate with a triphasic contrast-enhanced CT / MRI and specialist review.\n"
            "- Consider hepatology referral for further work-up.\n"
            "- Repeat tumour markers as clinically indicated."
        )
    else:
        rec = (
            "- No malignant pattern flagged by the models; routine surveillance as clinically advised.\n"
            "- Re-evaluate if clinical suspicion persists or markers change."
        )
    sections.append({"title": "Recommendations", "body": rec})

    summary = f"Hybrid AI assessment (clinical + CT): {final}."
    return {"summary": summary, "sections": sections}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def generate_report(req: dict) -> dict:
    """
    Build the final report. Uses the Gemini API when configured and falls back
    to the template otherwise, so the endpoint always returns a report.
    """
    bundle = build_findings_bundle(req)

    generated_by = "gemini"
    model: Optional[str] = MODEL_NAME
    try:
        out = _call_gemini(bundle)
    except Exception:
        out = _fallback_report(bundle)
        generated_by = "fallback"
        model = None

    return {
        "sections": out["sections"],
        "summary": out.get("summary", ""),
        "disclaimer": DISCLAIMER,
        "generated_by": generated_by,
        "model": model,
    }
