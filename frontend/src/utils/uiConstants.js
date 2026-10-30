// Placeholder patient details shown until a lab report is parsed.
export const defaultPatientInfo = {
  id: 'PT-2026-04851',
  name: 'Anonymous Patient',
  age: 58,
  sex: 'M',
  studyDate: '2026-05-07',
  modality: 'CT',
  series: 'Abdomen — Portal Venous Phase',
  sliceThickness: '2.5 mm',
  institution: 'Metro General Hospital',
  referringPhysician: 'Dr. R. Sharma',
};

// Labels for the Grad-CAM panel.
export const heatmapInfo = {
  method: 'Seg-Grad-CAM',
  targetLayer: 'encoder stage 3 (32×32)',
};

// Progress steps shown while a scan is analysed.
export const processingStages = [
  { label: 'Loading image', duration: 600 },
  { label: 'Preprocessing DICOM', duration: 800 },
  { label: 'Running segmentation model', duration: 1200 },
  { label: 'Computing tumor measurements', duration: 700 },
  { label: 'Generating Grad-CAM heatmap', duration: 900 },
  { label: 'Computing risk assessment', duration: 500 },
  { label: 'Finalizing report', duration: 400 },
];
