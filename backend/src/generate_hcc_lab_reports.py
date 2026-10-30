"""
Generator for synthetic hepatology lab report PDFs (used to demo the OCR step).

Each report has TUMOR MARKERS, HEPATIC FUNCTION PANEL and LIVER DISEASE
ETIOLOGY sections using the same schema and reference ranges as the clinical
dataset.

Patient values are drawn from the same distributions as the training data
(generate_hcc0 / generate_hcc1) and a candidate is kept only if both the
trained XGBoost model and the rule engine (app/blood_report.assess_markers)
agree with its intended label.

All names, institutions and values are fictional.

Usage:
    python -m src.generate_hcc_lab_reports   # 5 HCC + 5 no-HCC reports
"""

import os
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.generate_hcc_xgboost_dataset import generate_hcc0, generate_hcc1
from src.clinical_xgb_data import FEATURE_COLUMNS
from src.clinical_xgb_inference import ClinicalXGBInference
from app.blood_report import assess_markers

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
BLUE = "#1f4fd1"
INK = "#1a1f2b"
MUTED = "#6b7280"
BAND = "#eaf1fb"
BOX = "#f5f7fb"
LINE = "#e3e8f0"
GREEN = "#1a9f5a"
RED = "#d12b2b"
AMBER = "#c47a00"
FLAG_COLOR = {"N": GREEN, "H": RED, "L": AMBER}

CX = {"label": 0.045, "result": 0.50, "unit": 0.635, "ref": 0.735, "flag": 0.915}

# ---------------------------------------------------------------------------
# Synthetic identity pools (fictional Indian patient/physician names)
# ---------------------------------------------------------------------------
FIRST_NAMES_M = ["Arjun", "Rohan", "Vikram", "Suresh", "Anil", "Rajesh", "Karan",
                  "Amit", "Sanjay", "Deepak", "Manoj", "Ravi", "Ajay", "Vivek"]
FIRST_NAMES_F = ["Priya", "Anjali", "Neha", "Kavita", "Sunita", "Pooja", "Meera",
                  "Divya", "Shreya", "Lakshmi", "Anita", "Radhika", "Sneha", "Rekha"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Reddy", "Nair", "Gupta", "Mehta", "Patil",
              "Joshi", "Kulkarni", "Rao", "Chatterjee", "Menon", "Desai", "Kapoor"]
TITLES = {"M": "Mr.", "F": "Ms."}
PHYSICIAN_INITIALS = list("ABCDEFGHJKMNPRSTV")

# ---------------------------------------------------------------------------
# Reference ranges — must match src/generate_hcc_xgboost_dataset.py
# ---------------------------------------------------------------------------
REF = {
    "afp": {"lo": 0.0, "hi": 20.0, "unit": "ng/mL", "decimals": 1},
    "afp_l3": {"hi": 10.0, "unit": "%", "decimals": 1},           # "<10.0" upper-only
    "dcp": {"hi": 40.0, "unit": "mAU/mL", "decimals": 1},         # "<40.0" upper-only
    "albumin": {"lo": 3.5, "hi": 5.0, "unit": "g/dL", "decimals": 1},
    "bilirubin": {"lo": 0.1, "hi": 1.2, "unit": "mg/dL", "decimals": 2},
    "platelets": {"lo": 150, "hi": 450, "unit": "x10^9/L", "decimals": 0},
}


def _flag_range(value, lo, hi):
    if value < lo:
        return "L"
    if value > hi:
        return "H"
    return "N"


def _flag_upper(value, hi):
    return "H" if value >= hi else "N"


def _flag_categorical(value, normal_value):
    return "N" if value == normal_value else "H"


# ---------------------------------------------------------------------------
# Model-grounded patient sampling
# ---------------------------------------------------------------------------

def _score_batch_xgb(engine, df):
    """Vectorized XGBoost risk_probability for every row in df."""
    X = engine.preprocessor.transform(df[FEATURE_COLUMNS])
    X = pd.DataFrame(X, columns=engine.preprocessor.get_feature_names_out())
    return engine.model.predict_proba(X)[:, 1]


def _score_row_rule_engine(row):
    """Rule-engine score (app/blood_report.assess_markers) for one row."""
    result = assess_markers({
        "afp": float(row["AFP_ng_mL"]),
        "afp_l3": float(row["AFP_L3_percent"]),
        "dcp": float(row["DCP_mAU_mL"]),
        "albumin": float(row["Albumin_g_dL"]),
        "platelets": float(row["Platelet_count_x10e9_L"]),
        "total_bilirubin": float(row["Bilirubin_mg_dL"]),
    })
    return result["score"]


def _sample_confident_cohort(rng, hcc_label, n_needed, engine,
                              pool_size=500, max_rounds=6,
                              xgb_confident_lo=0.10, xgb_confident_hi=0.90,
                              rule_score_lo=1, rule_score_hi=3):
    """
    Draw candidates from the training-data generator and keep those that both
    the XGBoost model and the rule engine confidently assign to `hcc_label`.
    The pool grows each round until enough are found.
    """
    generator = generate_hcc1 if hcc_label == 1 else generate_hcc0
    accepted = []
    round_pool = pool_size

    for _ in range(max_rounds):
        df = generator(rng, n_target=round_pool)
        df = df.assign(_risk_proba=_score_batch_xgb(engine, df))
        df = df.assign(_rule_score=df.apply(_score_row_rule_engine, axis=1))

        if hcc_label == 1:
            keep = df[(df["_risk_proba"] >= xgb_confident_hi) & (df["_rule_score"] >= rule_score_hi)]
        else:
            keep = df[(df["_risk_proba"] <= xgb_confident_lo) & (df["_rule_score"] <= rule_score_lo)]

        accepted.append(keep)
        if sum(len(k) for k in accepted) >= n_needed:
            break
        round_pool *= 2  # not enough confident agreement yet — cast a wider net

    pool = pd.concat(accepted, ignore_index=True)
    if len(pool) < n_needed:
        raise RuntimeError(
            f"Could not find {n_needed} candidates where both the XGBoost model and "
            f"the rule engine confidently agree on HCC={hcc_label} after {max_rounds} "
            f"rounds (found {len(pool)}). Try loosening the confidence thresholds."
        )

    pool = pool.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    return pool.iloc[:n_needed].reset_index(drop=True)


def _row_to_patient(row):
    return {
        "hcc": int(row["HCC"]),
        "sex": row["Sex"],
        "age": int(row["Age"]),
        "afp": float(row["AFP_ng_mL"]),
        "afp_l3": float(row["AFP_L3_percent"]),
        "dcp": float(row["DCP_mAU_mL"]),
        "albumin": float(row["Albumin_g_dL"]),
        "bilirubin": float(row["Bilirubin_mg_dL"]),
        "platelets": float(row["Platelet_count_x10e9_L"]),
        "hbv": int(row["HBV_status"]),
        "hcv": int(row["HCV_status"]),
        "cirrhosis": int(row["Cirrhosis_status"]),
        "risk_proba": float(row["_risk_proba"]),
        "rule_score": int(row["_rule_score"]),
    }


def _random_identity(rng, sex, patient_id, base_date):
    first = rng.choice(FIRST_NAMES_M if sex == "M" else FIRST_NAMES_F)
    last = rng.choice(LAST_NAMES)
    name = f"{TITLES[sex]} {first} {last}"

    phys_last = rng.choice(LAST_NAMES)
    phys_initial = rng.choice(PHYSICIAN_INITIALS)
    physician = f"Dr. {phys_initial}. {phys_last}, MD"

    collect_offset_days = int(rng.integers(0, 60))
    collected = base_date - dt.timedelta(days=collect_offset_days,
                                          hours=int(rng.integers(0, 12)))
    reported = collected + dt.timedelta(hours=int(rng.integers(5, 30)))

    lab_id = f"SHD-{collected.year}-{int(rng.integers(100000, 999999))}"
    uhid = f"PT-{collected.year}-{int(rng.integers(1000, 99999))}"

    return {
        "name": name,
        "physician": physician,
        "collected": collected,
        "reported": reported,
        "lab_id": lab_id,
        "patient_id": uhid,
    }


def _build_rows(patient):
    afp = patient["afp"]
    afp_l3 = patient["afp_l3"]
    dcp = patient["dcp"]
    albumin = patient["albumin"]
    bilirubin = patient["bilirubin"]
    platelets = patient["platelets"]

    rows = [
        ("sec", "TUMOR MARKERS"),
        ("row", "Alpha-Fetoprotein (AFP)", f"{afp:.1f}", "ng/mL", "0.0 - 20.0",
         _flag_range(afp, REF["afp"]["lo"], REF["afp"]["hi"])),
        ("row", "AFP-L3 (% of total AFP)", f"{afp_l3:.1f}", "%", "< 10.0",
         _flag_upper(afp_l3, REF["afp_l3"]["hi"])),
        ("row", "Des-gamma-carboxy Prothrombin (DCP / PIVKA-II)", f"{dcp:.1f}", "mAU/mL", "< 40.0",
         _flag_upper(dcp, REF["dcp"]["hi"])),

        ("sec", "HEPATIC FUNCTION PANEL"),
        ("row", "Albumin", f"{albumin:.1f}", "g/dL", "3.5 - 5.0",
         _flag_range(albumin, REF["albumin"]["lo"], REF["albumin"]["hi"])),
        ("row", "Total Bilirubin", f"{bilirubin:.2f}", "mg/dL", "0.1 - 1.2",
         _flag_range(bilirubin, REF["bilirubin"]["lo"], REF["bilirubin"]["hi"])),
        ("row", "Platelet Count", f"{platelets:.0f}", "x10^9/L", "150 - 450",
         _flag_range(platelets, REF["platelets"]["lo"], REF["platelets"]["hi"])),

        ("sec", "LIVER DISEASE ETIOLOGY"),
        ("row", "Hepatitis B Surface Antigen (HBsAg)",
         "Reactive" if patient["hbv"] else "Non-reactive", "-", "Non-reactive",
         _flag_categorical("Reactive" if patient["hbv"] else "Non-reactive", "Non-reactive")),
        ("row", "Anti-HCV Antibody",
         "Reactive" if patient["hcv"] else "Non-reactive", "-", "Non-reactive",
         _flag_categorical("Reactive" if patient["hcv"] else "Non-reactive", "Non-reactive")),
        ("row", "Clinical Cirrhosis Status",
         "Present" if patient["cirrhosis"] else "Absent", "-", "Absent",
         _flag_categorical("Present" if patient["cirrhosis"] else "Absent", "Absent")),
    ]
    return rows


def _render_pdf(patient, identity, output_path):
    rows = _build_rows(patient)

    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def T(x, y, s, size=10, color=INK, weight="normal", ha="left", va="center"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                 ha=ha, va=va, family="DejaVu Sans", transform=ax.transAxes)

    # ---- Header / letterhead (fictional) ----
    T(0.045, 0.972, "SUNRISE HEPATOLOGY", size=15, color=BLUE, weight="bold")
    T(0.045, 0.955, "& DIAGNOSTICS LABORATORY", size=15, color=BLUE, weight="bold")
    T(0.045, 0.938, "NABL Accredited (Fictional)  ·  Dept. of Hepatology & Oncology Biochemistry",
      size=8.3, color=MUTED)
    T(0.955, 0.972, "47 Riverside Road, Pune 411001 (Fictional Address)", size=8, color=MUTED, ha="right")
    T(0.955, 0.959, "Ph: +91 20 4900 8800  ·  www.sunrisehepdiag.example", size=8, color=MUTED, ha="right")
    T(0.955, 0.946, "Synthetic demo lab — not a real institution", size=7.5, color=MUTED, ha="right")
    ax.add_patch(Rectangle((0.045, 0.925), 0.91, 0.004, color=BLUE, transform=ax.transAxes))

    T(0.5, 0.907, "L I V E R   T U M O R   M A R K E R   &   H E P A T I C   F U N C T I O N   R E P O R T",
      size=10.5, weight="bold", ha="center")

    # ---- Patient / accession block ----
    ax.add_patch(FancyBboxPatch((0.045, 0.80), 0.91, 0.088,
                 boxstyle="round,pad=0.006,rounding_size=0.012",
                 linewidth=1, edgecolor=LINE, facecolor=BOX, transform=ax.transAxes))
    patient_rows = [
        ("Patient Name", identity["name"], "Lab ID / Accession", identity["lab_id"]),
        ("Age / Sex", f"{patient['age']} Years / {'Male' if patient['sex'] == 'M' else 'Female'}",
         "Collected On", identity["collected"].strftime("%d-%b-%Y %H:%M")),
        ("Patient ID (UHID)", identity["patient_id"], "Reported On",
         identity["reported"].strftime("%d-%b-%Y %H:%M")),
        ("Referring Physician", identity["physician"], "Specimen", "Serum / Whole Blood (EDTA)"),
    ]
    y = 0.868
    for l1, v1, l2, v2 in patient_rows:
        T(0.065, y, l1, size=9, weight="bold", color="#374151")
        T(0.235, y, v1, size=9)
        T(0.52, y, l2, size=9, weight="bold", color="#374151")
        T(0.70, y, v2, size=9)
        y -= 0.019

    # ---- Table header ----
    top = 0.775
    ax.add_patch(Rectangle((0.045, top - 0.026), 0.91, 0.026, color=BLUE, transform=ax.transAxes))
    hy = top - 0.013
    T(CX["label"] + 0.01, hy, "INVESTIGATION", size=8.5, color="#fff", weight="bold")
    T(CX["result"], hy, "RESULT", size=8.5, color="#fff", weight="bold")
    T(CX["unit"], hy, "UNIT", size=8.5, color="#fff", weight="bold")
    T(CX["ref"], hy, "REFERENCE RANGE", size=8.5, color="#fff", weight="bold")
    T(CX["flag"], hy, "FLAG", size=8.5, color="#fff", weight="bold", ha="center")

    rh = 0.0265
    y = top - 0.026
    alt = False
    for r in rows:
        if r[0] == "sec":
            ax.add_patch(Rectangle((0.045, y - rh), 0.91, rh, color=BAND, transform=ax.transAxes))
            T(CX["label"] + 0.01, y - rh / 2, r[1], size=8.2, color=BLUE, weight="bold")
            alt = False
        else:
            _, label, result, unit, ref, flag = r
            if alt:
                ax.add_patch(Rectangle((0.045, y - rh), 0.91, rh, color="#fafbfe", transform=ax.transAxes))
            ax.add_patch(Rectangle((0.045, y - rh), 0.91, 0.0006, color=LINE, transform=ax.transAxes))
            cy = y - rh / 2
            T(CX["label"] + 0.01, cy, label, size=8.4)
            T(CX["result"], cy, result, size=8.7, weight="bold")
            T(CX["unit"], cy, unit, size=8.7, color="#374151")
            T(CX["ref"], cy, ref, size=8.7, color="#374151")
            T(CX["flag"], cy, flag, size=9, weight="bold", color=FLAG_COLOR[flag], ha="center")
            alt = not alt
        y -= rh

    # ---- Footer / signature ----
    fy = y - 0.05
    ax.add_patch(Rectangle((0.045, fy + 0.012), 0.91, 0.0006, color=LINE, transform=ax.transAxes))
    T(0.045, fy, "Report electronically generated. N = Normal, H = High, L = Low vs reference range.",
      size=7.5, color=MUTED)
    T(0.045, fy - 0.013, "For research / educational use — not a clinical diagnosis.",
      size=7.5, color=MUTED)
    T(0.955, fy, "Dr. R. Kulkarni, MD (Biochemistry)", size=9, color="#111827", weight="bold", ha="right")
    T(0.955, fy - 0.014, "Consultant Pathologist (Synthetic Signatory)", size=8, color=MUTED, ha="right")

    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def generate_patient_reports(n_hcc=5, n_no_hcc=5, output_dir="reports", seed=42,
                              xgb_confident_lo=0.10, xgb_confident_hi=0.90,
                              rule_score_lo=1, rule_score_hi=3):
    """
    Generate synthetic hepatology lab report PDFs — `n_hcc` labeled HCC=1 and
    `n_no_hcc` labeled HCC=0. Filenames encode the ground-truth label
    explicitly: lab_report_<UHID>_hcc{0,1}.pdf

    Values come from the same generative distributions that trained the
    deployed clinical XGBoost model (src/generate_hcc_xgboost_dataset.py),
    filtered to candidates where BOTH the actual trained model and the
    deterministic rule engine confidently agree with the intended label —
    see _sample_confident_cohort.

    Returns:
        List of file paths for the generated PDFs.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    engine = ClinicalXGBInference()
    engine.load_model()

    hcc1_rows = _sample_confident_cohort(
        rng, 1, n_hcc, engine,
        xgb_confident_lo=xgb_confident_lo, xgb_confident_hi=xgb_confident_hi,
        rule_score_lo=rule_score_lo, rule_score_hi=rule_score_hi,
    )
    hcc0_rows = _sample_confident_cohort(
        rng, 0, n_no_hcc, engine,
        xgb_confident_lo=xgb_confident_lo, xgb_confident_hi=xgb_confident_hi,
        rule_score_lo=rule_score_lo, rule_score_hi=rule_score_hi,
    )

    all_rows = pd.concat([hcc1_rows, hcc0_rows], ignore_index=True)
    all_rows = all_rows.sample(
        frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))
    ).reset_index(drop=True)

    base_date = dt.datetime(2026, 8, 5, 10, 0)
    paths = []
    for i, row in all_rows.iterrows():
        patient = _row_to_patient(row)
        identity = _random_identity(rng, patient["sex"], i, base_date)

        filename = f"lab_report_{identity['patient_id']}_hcc{patient['hcc']}.pdf"
        out_path = os.path.join(output_dir, filename)
        _render_pdf(patient, identity, out_path)

        print(f"[REPORT] {out_path}  |  HCC={patient['hcc']}  |  "
              f"xgb_risk={patient['risk_proba']:.3f}  |  rule_score={patient['rule_score']}  |  "
              f"patient={identity['name']} ({patient['age']}{patient['sex']})")

        paths.append(out_path)

    return paths


def main():
    output_dir = os.path.join("data", "hcc_lab_reports")

    # Remove reports from a previous run.
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith("lab_report_") and f.endswith(".pdf"):
                os.remove(os.path.join(output_dir, f))

    paths = generate_patient_reports(n_hcc=5, n_no_hcc=5, output_dir=output_dir, seed=42)
    print(f"\n[INFO] Generated {len(paths)} report(s) in {output_dir}:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
