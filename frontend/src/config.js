// Shared frontend configuration.
export const API_BASE = 'http://localhost:8000';

// Blood-report tumour markers we show in the editable review form,
// with display metadata. Keys match the backend BloodMarkers schema.
export const MARKER_FIELDS = [
  { key: 'afp',             label: 'Alpha-Fetoprotein (AFP)',        unit: 'ng/mL',   ref: '0 – 10' },
  { key: 'afp_l3',          label: 'AFP-L3 (% of total AFP)',        unit: '%',       ref: '< 10' },
  { key: 'dcp',             label: 'DCP / PIVKA-II',                 unit: 'mAU/mL',  ref: '< 40' },
  { key: 'albumin',         label: 'Albumin',                        unit: 'g/dL',    ref: '3.5 – 5.0' },
  { key: 'platelets',       label: 'Platelet Count',                 unit: '×10⁹/L',  ref: '150 – 400' },
  { key: 'total_bilirubin', label: 'Total Bilirubin',                unit: 'mg/dL',   ref: '0.1 – 1.2' },
];

// Review-form keys -> clinical XGBoost feature names (ClinicalXGBFeatures).
export const MARKER_TO_CLINICAL_KEY = {
  afp: 'AFP_ng_mL',
  afp_l3: 'AFP_L3_percent',
  dcp: 'DCP_mAU_mL',
  albumin: 'Albumin_g_dL',
  platelets: 'Platelet_count_x10e9_L',
  total_bilirubin: 'Bilirubin_mg_dL',
};
