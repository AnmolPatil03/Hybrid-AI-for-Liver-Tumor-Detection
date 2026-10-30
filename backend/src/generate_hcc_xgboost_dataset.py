"""
Synthetic clinical dataset generator for an XGBoost HCC-risk classifier.

THIS DATA IS SYNTHETIC. It is built programmatically from published
diagnostic-accuracy literature on AFP / AFP-L3% / DCP thresholds and typical
lab-value distributions for cirrhosis / viral hepatitis / HCC. It is for
model prototyping and education only and must never be presented as
real-patient data or as clinically validated.

Schema (one row per patient):
    Patient_ID, Age, Sex, HBV_status, HCV_status, Cirrhosis_status,
    Albumin_g_dL, Bilirubin_mg_dL, Platelet_count_x10e9_L,
    AFP_ng_mL, AFP_L3_percent, DCP_mAU_mL, HCC

Usage:
    python -m src.generate_hcc_xgboost_dataset
    python -m src.generate_hcc_xgboost_dataset --out data/hcc_xgboost/hcc_synthetic_dataset.xlsx
"""

import os
import argparse
import numpy as np
import pandas as pd

SEED = 42
N_PER_CLASS = 2500
OUTPUT_XLSX = os.path.join("data", "hcc_xgboost", "hcc_synthetic_dataset.xlsx")


def _lognormal(rng, median, sigma, size):
    """Lognormal draw parameterised by its median (so params are easy to reason about)."""
    return rng.lognormal(mean=np.log(median), sigma=sigma, size=size)


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


def _age_sex(rng, n, age_mean, age_sd, male_frac, age_lo=18, age_hi=90):
    age = _clip(rng.normal(age_mean, age_sd, n), age_lo, age_hi).round(0).astype(int)
    sex = np.where(rng.random(n) < male_frac, "M", "F")
    return age, sex


def _viral_status(rng, n, hbv_prob, hcv_prob, mutually_exclusive=True):
    hbv = rng.random(n) < hbv_prob
    if mutually_exclusive:
        hcv = (~hbv) & (rng.random(n) < hcv_prob)
    else:
        hcv = rng.random(n) < hcv_prob
    return hbv.astype(int), hcv.astype(int)


def _liver_panel(rng, n, cirrhotic):
    """Albumin / Bilirubin / Platelets, driven mainly by cirrhosis status."""
    cirrhotic = np.asarray(cirrhotic, dtype=bool)
    alb = np.where(
        cirrhotic,
        rng.normal(3.0, 0.5, n),
        rng.normal(4.3, 0.3, n),
    )
    albumin = _clip(alb, 1.8, 5.2).round(1)

    bili = np.where(
        cirrhotic,
        _lognormal(rng, 2.0, 0.6, n),
        _lognormal(rng, 0.6, 0.35, n),
    )
    bilirubin = _clip(bili, 0.2, 12.0).round(2)

    plt = np.where(
        cirrhotic,
        rng.normal(100, 45, n),
        rng.normal(250, 50, n),
    )
    platelets = _clip(plt, 15, 500).round(0).astype(int)

    return albumin, bilirubin, platelets


def _tumor_markers_baseline(rng, n):
    """Healthy/near-normal baseline draws for AFP, AFP-L3%, DCP."""
    afp = _clip(_lognormal(rng, 3.0, 0.5, n), 0.5, 15)
    afp_l3 = _clip(rng.normal(4.0, 2.0, n), 0, 100)
    dcp = _clip(_lognormal(rng, 22, 0.35, n), 5, 39)
    return afp, afp_l3, dcp


def generate_hcc0(rng, n_target=N_PER_CLASS):
    """Build the HCC=0 cohort as a set of overlapping clinical scenarios."""
    rows = []

    def n_for(frac):
        return max(1, round(n_target * frac))

    scenario_sizes = {
        "healthy": 0.15,
        "hbv_no_hcc": 0.12,
        "hcv_no_hcc": 0.12,
        "cirrhosis_no_hcc": 0.10,
        "hbv_cirrhosis_no_hcc": 0.08,
        "hcv_cirrhosis_no_hcc": 0.08,
        "abnormal_lft_no_hcc": 0.08,
        "elevated_afp_fp": 0.08,
        "elevated_dcp_fp": 0.07,
        "elevated_afpl3_fp": 0.06,
        "multi_marker_fp": 0.06,
    }
    sizes = {k: n_for(v) for k, v in scenario_sizes.items()}
    # Pad/trim so the sum matches n_target exactly.
    diff = n_target - sum(sizes.values())
    sizes["healthy"] += diff

    for scenario, n in sizes.items():
        if n <= 0:
            continue

        if scenario == "healthy":
            age, sex = _age_sex(rng, n, age_mean=40, age_sd=15, male_frac=0.50)
            hbv = np.zeros(n, dtype=int)
            hcv = np.zeros(n, dtype=int)
            cirrhosis = np.zeros(n, dtype=int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, afp_l3, dcp = _tumor_markers_baseline(rng, n)

        elif scenario == "hbv_no_hcc":
            age, sex = _age_sex(rng, n, age_mean=42, age_sd=14, male_frac=0.53)
            hbv = np.ones(n, dtype=int)
            hcv = np.zeros(n, dtype=int)
            cirrhosis = np.zeros(n, dtype=int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, afp_l3, dcp = _tumor_markers_baseline(rng, n)

        elif scenario == "hcv_no_hcc":
            age, sex = _age_sex(rng, n, age_mean=45, age_sd=13, male_frac=0.53)
            hbv = np.zeros(n, dtype=int)
            hcv = np.ones(n, dtype=int)
            cirrhosis = np.zeros(n, dtype=int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, afp_l3, dcp = _tumor_markers_baseline(rng, n)

        elif scenario == "cirrhosis_no_hcc":
            age, sex = _age_sex(rng, n, age_mean=55, age_sd=12, male_frac=0.55)
            hbv, hcv = _viral_status(rng, n, hbv_prob=0.25, hcv_prob=0.25)
            cirrhosis = np.ones(n, dtype=int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, afp_l3, dcp = _cirrhotic_marker_falsepos(rng, n)

        elif scenario == "hbv_cirrhosis_no_hcc":
            age, sex = _age_sex(rng, n, age_mean=53, age_sd=12, male_frac=0.55)
            hbv = np.ones(n, dtype=int)
            hcv = np.zeros(n, dtype=int)
            cirrhosis = np.ones(n, dtype=int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, afp_l3, dcp = _cirrhotic_marker_falsepos(rng, n)

        elif scenario == "hcv_cirrhosis_no_hcc":
            age, sex = _age_sex(rng, n, age_mean=56, age_sd=12, male_frac=0.55)
            hbv = np.zeros(n, dtype=int)
            hcv = np.ones(n, dtype=int)
            cirrhosis = np.ones(n, dtype=int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, afp_l3, dcp = _cirrhotic_marker_falsepos(rng, n)

        elif scenario == "abnormal_lft_no_hcc":
            age, sex = _age_sex(rng, n, age_mean=50, age_sd=14, male_frac=0.52)
            hbv, hcv = _viral_status(rng, n, hbv_prob=0.2, hcv_prob=0.2)
            cirrhosis = (rng.random(n) < 0.5).astype(int)
            albumin = _clip(rng.normal(2.9, 0.4, n), 1.8, 3.4).round(1)
            bilirubin = _clip(_lognormal(rng, 2.2, 0.5, n), 1.3, 8.0).round(2)
            platelets = _clip(rng.normal(180, 70, n), 30, 450).round(0).astype(int)
            afp, afp_l3, dcp = _tumor_markers_baseline(rng, n)

        elif scenario == "elevated_afp_fp":
            age, sex = _age_sex(rng, n, age_mean=48, age_sd=14, male_frac=0.52)
            hbv, hcv = _viral_status(rng, n, hbv_prob=0.25, hcv_prob=0.2)
            cirrhosis = (rng.random(n) < 0.4).astype(int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp = rng.uniform(20, 200, n)
            _, afp_l3, dcp = _tumor_markers_baseline(rng, n)

        elif scenario == "elevated_dcp_fp":
            age, sex = _age_sex(rng, n, age_mean=48, age_sd=14, male_frac=0.52)
            hbv, hcv = _viral_status(rng, n, hbv_prob=0.2, hcv_prob=0.25)
            cirrhosis = (rng.random(n) < 0.35).astype(int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, afp_l3, _ = _tumor_markers_baseline(rng, n)
            dcp = rng.uniform(40, 150, n)

        elif scenario == "elevated_afpl3_fp":
            age, sex = _age_sex(rng, n, age_mean=48, age_sd=14, male_frac=0.52)
            hbv, hcv = _viral_status(rng, n, hbv_prob=0.2, hcv_prob=0.2)
            cirrhosis = (rng.random(n) < 0.4).astype(int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, _, dcp = _tumor_markers_baseline(rng, n)
            afp_l3 = rng.uniform(10, 25, n)

        elif scenario == "multi_marker_fp":
            age, sex = _age_sex(rng, n, age_mean=50, age_sd=13, male_frac=0.53)
            hbv, hcv = _viral_status(rng, n, hbv_prob=0.25, hcv_prob=0.25)
            cirrhosis = (rng.random(n) < 0.6).astype(int)
            albumin, bilirubin, platelets = _liver_panel(rng, n, cirrhosis)
            afp, afp_l3, dcp = _tumor_markers_baseline(rng, n)
            # Force >=2 of the 3 markers elevated simultaneously.
            which = rng.integers(0, 3, n)  # marker left un-forced
            afp = np.where(which != 0, rng.uniform(21, 150, n), afp)
            afp_l3 = np.where(which != 1, rng.uniform(11, 22, n), afp_l3)
            dcp = np.where(which != 2, rng.uniform(41, 130, n), dcp)

        else:
            raise ValueError(scenario)

        rows.append(_assemble(
            n, age, sex, hbv, hcv, cirrhosis,
            albumin, bilirubin, platelets, afp, afp_l3, dcp, hcc=0,
        ))

    return pd.concat(rows, ignore_index=True)


def _cirrhotic_marker_falsepos(rng, n):
    """Tumor markers for cirrhotic-no-HCC patients: mostly near baseline, with a
    realistic minority crossing screening cutoffs from regeneration/inflammation."""
    afp, afp_l3, dcp = _tumor_markers_baseline(rng, n)
    afp_fp = rng.random(n) < 0.12
    afp = np.where(afp_fp, rng.uniform(40, 100, n), afp)
    dcp_fp = rng.random(n) < 0.11
    dcp = np.where(dcp_fp, rng.uniform(40, 90, n), dcp)
    afpl3_fp = rng.random(n) < 0.08
    afp_l3 = np.where(afpl3_fp, rng.uniform(10, 18, n), afp_l3)
    return afp, afp_l3, dcp


def generate_hcc1(rng, n_target=N_PER_CLASS):
    """
    Build the HCC=1 cohort. About 60% of patients are cirrhotic; all other
    attributes are drawn from conditional distributions so subgroups overlap.
    """
    n = n_target
    cirrhosis = (rng.random(n) < 0.60).astype(int)

    age = np.empty(n)
    sex = np.empty(n, dtype=object)
    hbv = np.empty(n, dtype=int)
    hcv = np.empty(n, dtype=int)

    # Etiology: ~25% HBV-only, ~25% HCV-only, remainder non-viral (NASH/alcohol/idiopathic).
    etio = rng.random(n)
    is_hbv = etio < 0.25
    is_hcv = (~is_hbv) & (etio < 0.50)
    is_nonviral = ~(is_hbv | is_hcv)

    hbv[is_hbv] = 1
    hcv[is_hbv] = 0
    hbv[is_hcv] = 0
    hcv[is_hcv] = 1
    hbv[is_nonviral] = 0
    hcv[is_nonviral] = 0

    age[is_hbv] = rng.normal(52, 10, is_hbv.sum())
    age[is_hcv] = rng.normal(60, 10, is_hcv.sum())
    age[is_nonviral] = rng.normal(65, 11, is_nonviral.sum())
    age = _clip(age, 25, 90).round(0).astype(int)

    male_frac = 0.72
    sex[:] = np.where(rng.random(n) < male_frac, "M", "F")

    # --- Liver panel driven by cirrhosis status, with cirrhotic-HCC skewed a bit
    # worse than plain cirrhosis (tumor burden adds hepatic stress). ---
    cirr_mask = cirrhosis.astype(bool)
    albumin = np.empty(n)
    bilirubin = np.empty(n)
    platelets = np.empty(n)

    nc = int(cirr_mask.sum())
    albumin[cirr_mask] = _clip(rng.normal(3.0, 0.5, nc), 1.8, 5.0)
    bilirubin[cirr_mask] = _clip(_lognormal(rng, 2.0, 0.65, nc), 0.3, 10.0)
    platelets[cirr_mask] = _clip(rng.normal(100, 45, nc), 15, 450)

    nn = n - nc
    ncirr_mask = ~cirr_mask
    albumin[ncirr_mask] = _clip(rng.normal(3.8, 0.4, nn), 2.2, 5.2)
    bilirubin[ncirr_mask] = _clip(_lognormal(rng, 1.0, 0.45, nn), 0.2, 8.0)
    platelets[ncirr_mask] = _clip(rng.normal(220, 60, nn), 40, 500)

    albumin = albumin.round(1)
    bilirubin = bilirubin.round(2)
    platelets = platelets.round(0).astype(int)

    # --- Disease-stage latent variable: "early" (~40%) vs "advanced" (~30%),
    # remainder mid-stage. Biases tumor-marker tier and (for advanced) liver panel. ---
    stage = rng.random(n)
    early = stage < 0.40
    advanced = stage >= 0.70
    mid = ~(early | advanced)

    # Advanced disease: push liver panel further down/up regardless of cirrhosis.
    albumin = np.where(advanced, _clip(albumin - rng.uniform(0.2, 0.7, n), 1.5, 5.2), albumin).round(1)
    bilirubin = np.where(advanced, _clip(bilirubin + rng.uniform(0.5, 3.0, n), 0.2, 15.0), bilirubin).round(2)
    platelets = np.where(advanced, _clip(platelets - rng.uniform(10, 60, n), 15, 500), platelets).round(0).astype(int)

    # --- Tumor markers ---
    # AFP: ~35-40% stay under the 20 ng/mL "AFP-negative HCC" cutoff regardless of
    # stage; of the marker-positive remainder, tier scales with stage.
    afp_negative = rng.random(n) < 0.38
    afp = np.empty(n)
    afp[afp_negative] = _clip(_lognormal(rng, 6, 0.7, afp_negative.sum()), 0.5, 19.9)

    pos = ~afp_negative
    pos_early = pos & early
    pos_mid = pos & mid
    pos_adv = pos & advanced
    afp[pos_early] = _clip(_lognormal(rng, 60, 0.9, pos_early.sum()), 20, 400)
    afp[pos_mid] = _clip(_lognormal(rng, 150, 1.0, pos_mid.sum()), 20, 2000)
    afp[pos_adv] = _clip(_lognormal(rng, 500, 1.1, pos_adv.sum()), 20, 6000)
    # Strongly-elevated subset (~20% overall) with a long tail into the thousands.
    strong_afp = rng.random(n) < 0.20
    afp[strong_afp] = _clip(_lognormal(rng, 800, 1.0, strong_afp.sum()), 400, 8000)
    afp = afp.round(1)

    # AFP-L3%: ~50% >=10%, ~25% strongly positive (>=35%); correlated with AFP positivity.
    afp_l3 = _clip(rng.normal(6.0, 4.0, n), 0, 100)
    l3_pos = rng.random(n) < 0.50
    afp_l3 = np.where(l3_pos, rng.uniform(10, 34, n), afp_l3)
    l3_strong = rng.random(n) < 0.25
    afp_l3 = np.where(l3_strong, rng.uniform(35, 90, n), afp_l3)
    afp_l3 = afp_l3.round(1)

    # DCP: ~55% >100 mAU/mL, many well above 250; advanced stage skews higher.
    dcp = _clip(_lognormal(rng, 35, 0.7, n), 5, 300)
    dcp_pos = rng.random(n) < 0.55
    dcp[dcp_pos] = _clip(_lognormal(rng, 220, 0.9, dcp_pos.sum()), 100, 4000)
    dcp[advanced] = _clip(dcp[advanced] * rng.uniform(1.2, 2.5, advanced.sum()), 5, 8000)
    dcp = dcp.round(1)

    # Ensure ~40-45% of HCC patients have at least one marker still below its
    # threshold (AFP<20, AFP-L3<10, DCP<40) — i.e. not all three positive at once.
    all_three_pos = (afp >= 20) & (afp_l3 >= 10) & (dcp >= 40)
    force_one_down = all_three_pos & (rng.random(n) < 0.5)
    which = rng.integers(0, 3, n)
    idx = np.where(force_one_down)[0]
    for i in idx:
        w = which[i]
        if w == 0:
            afp[i] = round(float(_clip(_lognormal(rng, 6, 0.6, 1), 0.5, 19.9)[0]), 1)
        elif w == 1:
            afp_l3[i] = round(float(rng.uniform(1, 9.5)), 1)
        else:
            dcp[i] = round(float(_clip(_lognormal(rng, 22, 0.3, 1), 5, 39)[0]), 1)

    return _assemble(
        n, age, sex, hbv, hcv, cirrhosis,
        albumin, bilirubin, platelets, afp, afp_l3, dcp, hcc=1,
    )


def _assemble(n, age, sex, hbv, hcv, cirrhosis, albumin, bilirubin, platelets,
              afp, afp_l3, dcp, hcc):
    return pd.DataFrame({
        "Age": np.asarray(age).astype(int),
        "Sex": np.asarray(sex),
        "HBV_status": np.asarray(hbv).astype(int),
        "HCV_status": np.asarray(hcv).astype(int),
        "Cirrhosis_status": np.asarray(cirrhosis).astype(int),
        "Albumin_g_dL": np.asarray(albumin).round(1),
        "Bilirubin_mg_dL": np.asarray(bilirubin).round(2),
        "Platelet_count_x10e9_L": np.asarray(platelets).astype(int),
        "AFP_ng_mL": np.asarray(afp).round(1),
        "AFP_L3_percent": _clip(np.asarray(afp_l3), 0, 100).round(1),
        "DCP_mAU_mL": np.asarray(dcp).round(1),
        "HCC": int(hcc),
    })


DATA_DICTIONARY = [
    ("Patient_ID", "-", "-", "Unique synthetic patient identifier (P00001-P05000)."),
    ("Age", "years", "18-90", "Patient age at time of labs/imaging."),
    ("Sex", "-", "M / F", "Biological sex."),
    ("HBV_status", "0/1", "-", "Chronic hepatitis B infection status (1 = positive)."),
    ("HCV_status", "0/1", "-", "Chronic hepatitis C infection status (1 = positive)."),
    ("Cirrhosis_status", "0/1", "-", "Presence of cirrhosis (1 = cirrhotic)."),
    ("Albumin_g_dL", "g/dL", "3.5-5.0", "Serum albumin; falls with worsening synthetic liver function."),
    ("Bilirubin_mg_dL", "mg/dL", "0.1-1.2", "Total serum bilirubin; rises with cirrhosis/biliary compromise."),
    ("Platelet_count_x10e9_L", "x10^9/L", "150-450", "Platelet count; lowered by portal hypertension/hypersplenism in cirrhosis."),
    ("AFP_ng_mL", "ng/mL", "<20", "Alpha-fetoprotein tumor marker. 200 ng/mL ~49% sens/~98% spec; 400 ng/mL ~32% sens/~99% spec for HCC."),
    ("AFP_L3_percent", "%", "<10", "Lens culinaris agglutinin-reactive AFP fraction. >=10% standard positivity cutoff; >=35% ~100% specificity."),
    ("DCP_mAU_mL", "mAU/mL", "<40", "Des-gamma-carboxy prothrombin (PIVKA-II). ~40 mAU/mL standard screening cutoff; ~250 mAU/mL high-specificity cutoff. Can be falsely elevated by vitamin K deficiency, cholestasis, or warfarin without HCC."),
    ("HCC", "0/1", "-", "Target label: hepatocellular carcinoma present (1) or absent (0)."),
]


def _print_summary(df):
    numeric_cols = [
        "Age", "Albumin_g_dL", "Bilirubin_mg_dL", "Platelet_count_x10e9_L",
        "AFP_ng_mL", "AFP_L3_percent", "DCP_mAU_mL",
    ]
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    for label, name in ((0, "HCC = 0"), (1, "HCC = 1")):
        sub = df[df["HCC"] == label]
        print(f"\n===== {name}  (n={len(sub)}) =====")
        desc = sub[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        print(desc.round(2).to_string())

    print("\n===== Categorical value counts (overall) =====")
    for col in ("Sex", "HBV_status", "HCV_status", "Cirrhosis_status"):
        print(f"\n-- {col} --")
        print(df[col].value_counts())

    print("\n===== Categorical value counts by HCC =====")
    for col in ("Sex", "HBV_status", "HCV_status", "Cirrhosis_status"):
        print(f"\n-- {col} by HCC --")
        print(df.groupby("HCC")[col].value_counts(normalize=True).round(3))

    afp_neg_hcc = ((df["HCC"] == 1) & (df["AFP_ng_mL"] < 20)).sum()
    n_hcc = (df["HCC"] == 1).sum()
    print(f"\nAFP-negative HCC (AFP<20, HCC=1): {afp_neg_hcc}/{n_hcc} "
          f"({afp_neg_hcc / n_hcc:.1%})")

    dcp_fp_cirr = ((df["HCC"] == 0) & (df["Cirrhosis_status"] == 1) & (df["DCP_mAU_mL"] >= 40)).sum()
    n_cirr0 = ((df["HCC"] == 0) & (df["Cirrhosis_status"] == 1)).sum()
    if n_cirr0:
        print(f"DCP false-positive rate in cirrhotic non-HCC (DCP>=40): {dcp_fp_cirr}/{n_cirr0} "
              f"({dcp_fp_cirr / n_cirr0:.1%})")


def build_dataset(seed=SEED, n_per_class=N_PER_CLASS):
    rng = np.random.default_rng(seed)

    df0 = generate_hcc0(rng, n_per_class)
    df1 = generate_hcc1(rng, n_per_class)

    df = pd.concat([df0, df1], ignore_index=True)
    assert len(df) == 2 * n_per_class

    # Shuffle rows so scenario grouping doesn't leak an ordinal pattern.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df.insert(0, "Patient_ID", [f"P{i+1:05d}" for i in range(len(df))])

    col_order = [
        "Patient_ID", "Age", "Sex", "HBV_status", "HCV_status", "Cirrhosis_status",
        "Albumin_g_dL", "Bilirubin_mg_dL", "Platelet_count_x10e9_L",
        "AFP_ng_mL", "AFP_L3_percent", "DCP_mAU_mL", "HCC",
    ]
    df = df[col_order]

    assert df["HCC"].sum() == n_per_class
    assert (df["HCC"] == 0).sum() == n_per_class
    assert not df.duplicated(subset=[c for c in col_order if c != "Patient_ID"]).any()
    assert df.isna().sum().sum() == 0
    assert (df["Platelet_count_x10e9_L"] > 0).all()
    assert (df["Bilirubin_mg_dL"] > 0).all()

    return df


def write_xlsx(df, output_xlsx=OUTPUT_XLSX):
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    dict_df = pd.DataFrame(
        DATA_DICTIONARY, columns=["Column", "Unit", "Normal_Range", "Clinical_Meaning"]
    )
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="patients", index=False)
        dict_df.to_excel(writer, sheet_name="data_dictionary", index=False)
    return output_xlsx


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic HCC XGBoost dataset.")
    ap.add_argument("--out", default=OUTPUT_XLSX)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    args = ap.parse_args()

    df = build_dataset(seed=args.seed, n_per_class=args.n_per_class)
    _print_summary(df)

    out = write_xlsx(df, args.out)
    print(f"\n[INFO] Wrote {len(df)} rows to {out}")
    print("[NOTE] This dataset is SYNTHETIC — for model prototyping/education only, "
          "not derived from or validated against real patients.")


if __name__ == "__main__":
    main()
