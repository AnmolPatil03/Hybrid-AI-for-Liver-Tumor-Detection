"""
Blood report OCR and rule-based triage.

    1. OCR the uploaded report (PDF or image) and parse tumor markers,
       demographics and the clinical model's features   [extract_from_upload]
    2. The user reviews and corrects the values in the UI.
    3. A rule engine scores the confirmed markers against clinical
       thresholds                                          [assess_markers]

The rule engine is deterministic so its decision is easy to explain; the
XGBoost model (src/clinical_xgb_inference.py) runs alongside it.

Requires the Tesseract binary:
    macOS:    brew install tesseract
    Ubuntu:   apt-get install tesseract-ocr
    Windows:  https://github.com/UB-Mannheim/tesseract/wiki, then add it to
              PATH or set TESSERACT_CMD to the full path of tesseract.exe.
"""

import io
import os
import re
from typing import Optional

_tesseract_cmd = os.environ.get("TESSERACT_CMD")
if _tesseract_cmd:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


# --------------------------------------------------------------------------- #
# OCR text extraction
# --------------------------------------------------------------------------- #

def _pdf_to_images(pdf_bytes: bytes, dpi: int = 200):
    """Render each page of a PDF to a PIL image using PyMuPDF (no external poppler)."""
    import fitz  # PyMuPDF
    from PIL import Image

    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    return images


def _ocr_images(images) -> str:
    """Run Tesseract over a list of PIL images and concatenate the text."""
    import pytesseract
    from pytesseract import TesseractNotFoundError

    try:
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed on the host. "
            "Install it with 'brew install tesseract' (macOS), "
            "'apt-get install tesseract-ocr' (Ubuntu), or on Windows via "
            "https://github.com/UB-Mannheim/tesseract/wiki and set the "
            "TESSERACT_CMD env var to tesseract.exe's full path if it "
            "isn't on PATH."
        ) from exc


def extract_text(file_bytes: bytes, content_type: Optional[str], filename: Optional[str]) -> str:
    """
    Extract raw text from an uploaded blood report (PDF or image) via OCR.

    Detects PDFs by content-type, extension, or the '%PDF' magic bytes so a
    mislabelled upload is still handled correctly.
    """
    is_pdf = (
        (content_type == "application/pdf")
        or (filename or "").lower().endswith(".pdf")
        or file_bytes[:5] == b"%PDF-"
    )

    if is_pdf:
        images = _pdf_to_images(file_bytes)
    else:
        from PIL import Image
        images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

    return _ocr_images(images)


# --------------------------------------------------------------------------- #
# Marker parsing
# --------------------------------------------------------------------------- #

# A numeric token: 4, 4.2, 245, 1.0 ... (optionally signed)
_NUMBER = r"[-+]?\d+(?:\.\d+)?"

# Each marker: the key we store it under, and the regexes that identify its row.
# `_afp` uses a negative lookahead so it does NOT match the "AFP-L3" row.
_MARKER_PATTERNS = [
    ("afp",     [r"alpha[-\s]*fetoprotein", r"\bAFP\b(?!\s*[-\s]?\s*L3)"]),
    ("afp_l3",  [r"AFP\s*[-\s]?\s*L3", r"fetoprotein[-\s]*L3"]),
    ("dcp",     [r"des[-\s]*gamma", r"\bDCP\b", r"PIVKA"]),
    # Supporting markers — parsed if present, used only to nudge the score.
    ("albumin",          [r"\balbumin\b"]),
    ("platelets",        [r"platelet"]),
    ("total_bilirubin",  [r"total\s+bilirubin"]),
]


def _first_number_after(line: str, match_end: int) -> Optional[float]:
    """Return the first numeric token appearing after `match_end` on the line."""
    m = re.search(_NUMBER, line[match_end:])
    return float(m.group()) if m else None


# --------------------------------------------------------------------------- #
# Patient demographics parsing
# --------------------------------------------------------------------------- #

# Labels that mark the start of the *next* field on a line, used to bound where
# a captured value ends. Reports typically place two label/value pairs per line
# (e.g. "Patient Name ... Lab ID ..."), so we stop a capture at the next label.
_PT_STOP = (
    r"(?:Lab\s*ID|Accession|Patient\s*ID|UHID|\bAge\b|\bSex\b|Collected|"
    r"Reported|Referring|Physician|Specimen|Sample|\bDate\b|$)"
)


def _grab_after(text: str, label_pat: str, stop_pat: str = _PT_STOP) -> Optional[str]:
    """
    Find `label_pat` in the OCR text and return the value that follows it on the
    same line, bounded by the next known label (`stop_pat`). Parenthetical notes
    like "(UHID)" are stripped from the result.
    """
    m = re.search(
        label_pat + r"\s*[:\-]?\s*(.+?)\s*(?=" + stop_pat + r")",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        return None
    val = re.sub(r"\([^)]*\)", "", m.group(1))   # drop "(UHID)" etc.
    val = val.strip(" :\t|/-")
    return val or None


def parse_patient_info(text: str) -> dict:
    """
    Best-effort extraction of patient demographics from OCR text so the imaging
    UI can show the same patient the report belongs to. Every field is optional;
    anything that cannot be read comes back as None and the frontend keeps its
    default placeholder.
    """
    name = _grab_after(text, r"Patient\s+Name")

    # Prefer the UHID / Patient ID over the lab accession number.
    pid = _grab_after(text, r"Patient\s*ID(?:\s*\(?\s*UHID\s*\)?)?")
    if not pid:
        pid = _grab_after(text, r"\bUHID\b")
    if pid:
        m = re.search(r"[A-Za-z]{0,4}[-\s]?\d[\w-]*", pid)  # ID-like token
        pid = m.group().replace(" ", "") if m else pid

    age = None
    sex = None
    age_sex = _grab_after(text, r"Age\s*/?\s*Sex") or _grab_after(text, r"\bAge\b")
    if age_sex:
        am = re.search(r"\d{1,3}", age_sex)
        if am:
            age = int(am.group())
        sm = re.search(r"\b(male|female|m|f)\b", age_sex, flags=re.IGNORECASE)
        if sm:
            sex = "M" if sm.group(1).lower() in ("male", "m") else "F"

    # Reports that print Sex as its own field ("Sex: Female").
    if sex is None:
        sex_val = _grab_after(text, r"\bSex\b")
        if sex_val:
            sm = re.search(r"\b(male|female|m|f)\b", sex_val, flags=re.IGNORECASE)
            if sm:
                sex = "M" if sm.group(1).lower() in ("male", "m") else "F"

    # Study date: prefer the collection date, fall back to the reported date.
    study_date = None
    date_val = _grab_after(text, r"Collected\s*On") or _grab_after(text, r"Reported\s*On")
    if date_val:
        dm = re.search(r"\d{1,2}[-/\s][A-Za-z0-9]{2,}[-/\s]\d{2,4}", date_val)
        study_date = dm.group() if dm else date_val.split()[0]

    return {
        "name": name,
        "patient_id": pid,
        "age": age,
        "sex": sex,
        "study_date": study_date,
    }


# --------------------------------------------------------------------------- #
# Clinical feature extraction for the XGBoost model
# --------------------------------------------------------------------------- #
#
# For each model feature: label synonyms, unit and parse type. Age and Sex
# come from parse_patient_info(). Fields that can't be found are returned as
# None; they are never guessed here.
CLINICAL_FIELD_SPECS = {
    "AFP_ng_mL": {
        "synonyms": ["Alpha-Fetoprotein (AFP)", "Alpha Fetoprotein", "AFP"],
        "patterns": [r"alpha[-\s]*fetoprotein", r"\bAFP\b(?!\s*[-\s]?\s*L3)"],
        "unit": "ng/mL", "kind": "numeric",
    },
    "AFP_L3_percent": {
        "synonyms": ["AFP-L3 (% of total AFP)", "AFP-L3", "AFP L3%"],
        "patterns": [r"AFP\s*[-\s]?\s*L3", r"fetoprotein[-\s]*L3"],
        "unit": "%", "kind": "numeric",
    },
    "DCP_mAU_mL": {
        "synonyms": ["Des-gamma-carboxy Prothrombin (DCP / PIVKA-II)", "DCP", "PIVKA-II"],
        "patterns": [r"des[-\s]*gamma", r"\bDCP\b", r"PIVKA"],
        "unit": "mAU/mL", "kind": "numeric",
    },
    "Albumin_g_dL": {
        "synonyms": ["Albumin"],
        "patterns": [r"\balbumin\b"],
        "unit": "g/dL", "kind": "numeric",
    },
    "Bilirubin_mg_dL": {
        "synonyms": ["Total Bilirubin", "Bilirubin"],
        "patterns": [r"total\s+bilirubin", r"\bbilirubin\b"],
        "unit": "mg/dL", "kind": "numeric",
    },
    "Platelet_count_x10e9_L": {
        "synonyms": ["Platelet Count", "Platelets"],
        "patterns": [r"platelet"],
        "unit": "x10^9/L", "kind": "numeric",
    },
    "Age": {
        "synonyms": ["Age", "Age / Sex"],
        "unit": "years", "kind": "demographic",
    },
    "Sex": {
        "synonyms": ["Sex", "Age / Sex"],
        "unit": None, "kind": "demographic",
    },
    "HBV_status": {
        "synonyms": ["Hepatitis B Surface Antigen (HBsAg)", "HBsAg"],
        "patterns": [r"hep(?:atitis)?\s*B\s*(?:surface\s*antigen)?", r"\bHBsAg\b"],
        "unit": None, "kind": "reactive",
    },
    "HCV_status": {
        "synonyms": ["Anti-HCV Antibody", "Anti-HCV", "HCV Antibody"],
        "patterns": [r"anti[-\s]*HCV", r"hepatitis\s*C"],
        "unit": None, "kind": "reactive",
    },
    "Cirrhosis_status": {
        "synonyms": ["Clinical Cirrhosis Status", "Cirrhosis"],
        "patterns": [r"cirrhosis"],
        "unit": None, "kind": "present",
    },
}


def _first_reactive_after(line: str, match_end: int) -> Optional[int]:
    """
    Parse Reactive/Non-reactive after `match_end`. Takes the leftmost match
    because the reference-range column on the same line also contains
    "Non-reactive".
    """
    tail = line[match_end:]
    m = re.search(r"(non[-\s]*)?\breactive\b", tail, flags=re.IGNORECASE)
    if not m:
        return None
    return 0 if m.group(1) else 1


def _first_presence_after(line: str, match_end: int) -> Optional[int]:
    """Parse Present/Absent after `match_end` (leftmost match, as above)."""
    tail = line[match_end:]
    m = re.search(r"\b(present|absent)\b", tail, flags=re.IGNORECASE)
    if not m:
        return None
    return 1 if m.group(1).lower() == "present" else 0


def parse_clinical_features(text: str, patient_info: Optional[dict] = None) -> dict:
    """
    Parse the 11 clinical model features from OCR text. Fields that can't be
    found are None.

    Args:
        text: raw OCR text.
        patient_info: optional parse_patient_info(text) result.
    """
    results = {key: None for key in CLINICAL_FIELD_SPECS}
    lines = text.splitlines()

    for key, spec in CLINICAL_FIELD_SPECS.items():
        if spec["kind"] == "demographic":
            continue

        for line in lines:
            matched = False
            for pat in spec["patterns"]:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if not m:
                    continue
                if spec["kind"] == "numeric":
                    value = _first_number_after(line, m.end())
                elif spec["kind"] == "reactive":
                    value = _first_reactive_after(line, m.end())
                elif spec["kind"] == "present":
                    value = _first_presence_after(line, m.end())
                else:
                    value = None
                if value is not None:
                    results[key] = value
                    matched = True
                    break
            if matched:
                break

    info = patient_info if patient_info is not None else parse_patient_info(text)
    results["Age"] = info.get("age")
    results["Sex"] = info.get("sex")

    return results


def parse_markers(text: str) -> dict:
    """
    Parse tumour-marker (and supporting) values out of OCR text.

    Returns a dict keyed by marker; each value is the parsed float or None if the
    marker's row was not found. The result on a lab report line is the first
    number after the marker name (before the unit / reference range), e.g.

        "Alpha-Fetoprotein (AFP)   4.2   ng/mL   0.0 - 10.0   N"
                                    ^-- parsed value
    """
    results = {key: None for key, _ in _MARKER_PATTERNS}
    lines = text.splitlines()

    for key, patterns in _MARKER_PATTERNS:
        for line in lines:
            matched = False
            for pat in patterns:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if m:
                    value = _first_number_after(line, m.end())
                    if value is not None:
                        results[key] = value
                        matched = True
                        break
            if matched:
                break

    return results


# --------------------------------------------------------------------------- #
# Rule engine
# --------------------------------------------------------------------------- #

# Reference limits used for triage.
_AFP_NORMAL = 10.0        # ng/mL
_AFP_HIGH = 200.0         # ng/mL — strongly suggestive of HCC
_AFP_L3_HIGH = 10.0       # %
_DCP_NORMAL = 40.0        # mAU/mL
_DCP_HIGH = 150.0         # mAU/mL
_ALBUMIN_LOW = 3.5        # g/dL
_PLATELETS_LOW = 150.0    # x10^9/L
_BILIRUBIN_HIGH = 1.2     # mg/dL

# Score at/above this => recommend a CT scan.
_CT_THRESHOLD = 2


def assess_markers(values: dict) -> dict:
    """
    Score confirmed marker values against clinical thresholds.

    `values` may contain: afp, afp_l3, dcp (primary) and optionally albumin,
    platelets, total_bilirubin (supporting). Missing values are simply skipped.

    Returns a dict with the numeric score, risk level, per-marker flags, a
    plain-language interpretation, and `recommend_ct` — the flag the frontend
    uses to decide whether to prompt for a CT-scan upload.
    """
    score = 0
    flags = {}          # marker -> "normal" | "borderline" | "high" | "low"
    reasons = []

    def g(key):
        v = values.get(key)
        return v if isinstance(v, (int, float)) else None

    afp, afp_l3, dcp = g("afp"), g("afp_l3"), g("dcp")

    # --- Primary tumour markers ---
    if afp is not None:
        if afp > _AFP_HIGH:
            score += 2; flags["afp"] = "high"
            reasons.append(f"AFP markedly elevated ({afp:g} ng/mL, ref ≤{_AFP_NORMAL:g})")
        elif afp > _AFP_NORMAL:
            score += 1; flags["afp"] = "borderline"
            reasons.append(f"AFP mildly elevated ({afp:g} ng/mL, ref ≤{_AFP_NORMAL:g})")
        else:
            flags["afp"] = "normal"

    if afp_l3 is not None:
        if afp_l3 > _AFP_L3_HIGH:
            score += 1; flags["afp_l3"] = "high"
            reasons.append(f"AFP-L3 elevated ({afp_l3:g}%, ref <{_AFP_L3_HIGH:g}%)")
        else:
            flags["afp_l3"] = "normal"

    if dcp is not None:
        if dcp > _DCP_HIGH:
            score += 2; flags["dcp"] = "high"
            reasons.append(f"DCP/PIVKA-II markedly elevated ({dcp:g} mAU/mL, ref <{_DCP_NORMAL:g})")
        elif dcp > _DCP_NORMAL:
            score += 1; flags["dcp"] = "borderline"
            reasons.append(f"DCP/PIVKA-II elevated ({dcp:g} mAU/mL, ref <{_DCP_NORMAL:g})")
        else:
            flags["dcp"] = "normal"

    # --- Supporting markers (each nudges the score by 1) ---
    albumin, platelets, bili = g("albumin"), g("platelets"), g("total_bilirubin")
    if albumin is not None:
        if albumin < _ALBUMIN_LOW:
            score += 1; flags["albumin"] = "low"
            reasons.append(f"Low albumin ({albumin:g} g/dL) — possible impaired synthesis")
        else:
            flags["albumin"] = "normal"
    if platelets is not None:
        if platelets < _PLATELETS_LOW:
            score += 1; flags["platelets"] = "low"
            reasons.append(f"Low platelets ({platelets:g} ×10⁹/L) — possible portal hypertension")
        else:
            flags["platelets"] = "normal"
    if bili is not None:
        if bili > _BILIRUBIN_HIGH:
            score += 1; flags["total_bilirubin"] = "high"
            reasons.append(f"Elevated total bilirubin ({bili:g} mg/dL)")
        else:
            flags["total_bilirubin"] = "normal"

    # --- Decision ---
    if any(v is not None for v in (afp, afp_l3, dcp)):
        recommend_ct = score >= _CT_THRESHOLD
    else:
        # No primary marker could be read — cannot triage safely.
        recommend_ct = None

    if recommend_ct is None:
        risk_level = "unknown"
        interpretation = (
            "No tumour-marker values could be read from the report. Please enter "
            "AFP, AFP-L3% and DCP manually before an assessment can be made."
        )
    elif recommend_ct:
        risk_level = "suspicious"
        interpretation = (
            "Tumour markers are abnormal (" + "; ".join(reasons) + "). Findings are "
            "concerning for possible hepatocellular carcinoma. Cross-sectional "
            "imaging (CT/MRI) is recommended — please upload a CT scan for "
            "further analysis."
        )
    else:
        risk_level = "low"
        interpretation = (
            "Tumour markers are within normal limits. No biochemical evidence "
            "suggestive of hepatocellular carcinoma; routine surveillance as "
            "clinically advised."
        )

    return {
        "score": score,
        "risk_level": risk_level,
        "recommend_ct": recommend_ct,
        "flags": flags,
        "reasons": reasons,
        "interpretation": interpretation,
        "values": {k: values.get(k) for k, _ in _MARKER_PATTERNS},
    }


def extract_from_upload(file_bytes: bytes, content_type: Optional[str], filename: Optional[str]) -> dict:
    """OCR an upload and parse markers, demographics and clinical features."""
    text = extract_text(file_bytes, content_type, filename)
    patient = parse_patient_info(text)
    return {
        "markers": parse_markers(text),
        "patient": patient,
        "clinical_features": parse_clinical_features(text, patient_info=patient),
        "raw_text": text,
    }
