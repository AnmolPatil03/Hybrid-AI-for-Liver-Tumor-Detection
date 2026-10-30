// Shared tumour-measurement derivation, used by both the on-screen
// MeasurementsPanel and the PDF in FinalReportPanel so the numbers always match.
//
// Units: the API returns pixels. PNG slices carry no DICOM PixelSpacing, so the
// mm values use the assumed spacing below and are approximate.

export const PIXEL_SPACING_MM = 0.7;   // assumed in-plane spacing (typical abdo CT)
export const SLICE_THICKNESS_MM = 2.5; // assumed slice thickness
export const SPACING_ASSUMED = true;   // flag so the UI can show a caveat

/**
 * Derive tumour measurements from an /analyze result.
 * Returns null when there is no tumour / bounding box to measure.
 */
export function deriveMeasurements(analysisResult) {
  const bbox = analysisResult?.tumor_bbox;
  if (!bbox || !analysisResult?.has_tumor) return null;

  const px = analysisResult.tumor_area_pixels || 0;
  const s = PIXEL_SPACING_MM;

  const widthMm = bbox.width * s;
  const heightMm = bbox.height * s;
  const areaMm2 = px * s * s;
  // Single-slice volume estimate: segmented area × slice thickness.
  const volumeMm3 = areaMm2 * SLICE_THICKNESS_MM;
  const perimeterMm = (widthMm + heightMm) * 2;
  // Circularity is dimensionless — compute it entirely in pixel space so units
  // stay consistent: 4π·area / perimeter².
  const perimeterPx = (bbox.width + bbox.height) * 2;
  const circularity = (4 * Math.PI * (px || 1)) / (perimeterPx ** 2);

  return {
    widthMm, heightMm, areaMm2, volumeMm3, perimeterMm, circularity,
    areaPx: px,
    areaPct: analysisResult.tumor_area_percentage ?? null,
    spacingAssumed: SPACING_ASSUMED,
  };
}
