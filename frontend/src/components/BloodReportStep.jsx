import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FileText, Upload, CloudUpload, X, Loader2, FlaskConical,
  CheckCircle2, AlertTriangle, ArrowRight, ChevronDown, Activity, GitBranch,
} from 'lucide-react';
import { API_BASE, MARKER_FIELDS, MARKER_TO_CLINICAL_KEY } from '../config';
import DecisionTreeModal from './DecisionTreeModal';

/**
 * Steps 1-2: upload a lab report, OCR the markers, let the user correct them,
 * then call the rule engine (/blood/assess) and the XGBoost model
 * (/clinical/predict). The rule engine's `recommend_ct` controls the
 * "Proceed to CT" button; the XGBoost result is shown alongside it.
 *
 * Props:
 *   onAssessed(assessment)         — called when /blood/assess returns
 *   onPatientInfo(patient)         — called with demographics parsed from the report
 *   onClinicalRisk(risk)           — called when /clinical/predict returns
 *   onProceedToCT(clinicalValues)  — called when user proceeds to step 3
 */
export default function BloodReportStep({ onAssessed, onPatientInfo, onClinicalRisk, onProceedToCT }) {
  const [fileName, setFileName] = useState(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractError, setExtractError] = useState(null);
  const [rawText, setRawText] = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const [markers, setMarkers] = useState(null); // editable values
  const [isAssessing, setIsAssessing] = useState(false);
  const [assessError, setAssessError] = useState(null);
  const [assessment, setAssessment] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [noMarkers, setNoMarkers] = useState(false);
  const [ocrClinicalFeatures, setOcrClinicalFeatures] = useState(null); // Age/Sex/HBV/HCV/Cirrhosis from OCR
  const [clinicalRisk, setClinicalRisk] = useState(null);
  const [clinicalRiskError, setClinicalRiskError] = useState(null);
  const [showTreeModal, setShowTreeModal] = useState(false);
  const inputRef = useRef(null);

  // Step 1: pick a file and preview it — no OCR runs yet.
  const selectFile = useCallback((f) => {
    if (!f) return;
    // Only PDF / PNG / JPG are supported (drag-drop bypasses the picker filter).
    const supported = /^(application\/pdf|image\/(png|jpe?g))$/i.test(f.type)
      || /\.(pdf|png|jpe?g)$/i.test(f.name);
    if (!supported) {
      setExtractError(`“${f.name}” isn’t a supported file. Please upload the blood report as a PDF, PNG, or JPG.`);
      return;
    }
    setFile(f);
    setFileName(f.name);
    setPreviewUrl(URL.createObjectURL(f));
    setExtractError(null);
    setNoMarkers(false);
    setMarkers(null);
    setAssessment(null);
    setRawText(null);
    setOcrClinicalFeatures(null);
    setClinicalRisk(null);
    setClinicalRiskError(null);
    onAssessed?.(null);
    onPatientInfo?.(null);
    onClinicalRisk?.(null);
  }, [onAssessed, onPatientInfo, onClinicalRisk]);

  const clearFile = useCallback(() => {
    setFile(null);
    setFileName(null);
    setPreviewUrl(null);
    setNoMarkers(false);
    setMarkers(null);
    setAssessment(null);
    setRawText(null);
    setExtractError(null);
    setOcrClinicalFeatures(null);
    setClinicalRisk(null);
    setClinicalRiskError(null);
    onAssessed?.(null);
    onPatientInfo?.(null);
    onClinicalRisk?.(null);
  }, [onAssessed, onPatientInfo, onClinicalRisk]);

  // Free the preview object URL when it changes or the component unmounts.
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  // Step 2: user clicks "Extract values" — OCR the selected report.
  const handleExtract = useCallback(async () => {
    if (!file) return;
    setIsExtracting(true);
    setExtractError(null);
    setNoMarkers(false);
    setAssessment(null);
    setMarkers(null);
    setClinicalRisk(null);
    setClinicalRiskError(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_BASE}/blood/extract`, { method: 'POST', body: fd });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Extraction failed (${res.status})`);
      }
      const data = await res.json();
      // Seed the editable form with parsed values (blank where OCR found nothing).
      const seeded = {};
      MARKER_FIELDS.forEach((f) => {
        const v = data.markers?.[f.key];
        seeded[f.key] = v === null || v === undefined ? '' : String(v);
      });
      setMarkers(seeded);
      // Nothing recognised → likely not a (readable) tumour-marker report.
      setNoMarkers(!MARKER_FIELDS.some((f) => data.markers?.[f.key] != null));
      setRawText(data.raw_text || '');
      // Age/Sex/HBV/HCV/Cirrhosis are passed to the XGBoost model as read by OCR.
      setOcrClinicalFeatures(data.clinical_features || null);
      onPatientInfo?.(data.patient || null);
    } catch (err) {
      setExtractError(err.message);
    } finally {
      setIsExtracting(false);
    }
  }, [file, onPatientInfo]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    selectFile(e.dataTransfer.files?.[0]);
  }, [selectFile]);

  const updateMarker = (key, val) => {
    // allow digits, dot, minus
    if (/^-?\d*\.?\d*$/.test(val)) setMarkers((m) => ({ ...m, [key]: val }));
  };

  const toPayload = () => {
    const p = {};
    MARKER_FIELDS.forEach((f) => {
      if (markers[f.key] !== '' && markers[f.key] !== undefined) {
        const n = parseFloat(markers[f.key]);
        if (!Number.isNaN(n)) p[f.key] = n;
      }
    });
    return p;
  };

  // Payload for /clinical/predict: reviewed markers renamed to model feature
  // names, plus the OCR demographics/etiology fields.
  const toClinicalFeaturesPayload = () => {
    const markerPayload = toPayload();
    const p = {};
    Object.entries(MARKER_TO_CLINICAL_KEY).forEach(([markerKey, clinicalKey]) => {
      if (markerPayload[markerKey] !== undefined) p[clinicalKey] = markerPayload[markerKey];
    });
    if (ocrClinicalFeatures) {
      ['Age', 'Sex', 'HCV_status', 'HBV_status', 'Cirrhosis_status'].forEach((k) => {
        if (ocrClinicalFeatures[k] !== undefined && ocrClinicalFeatures[k] !== null) {
          p[k] = ocrClinicalFeatures[k];
        }
      });
    }
    return p;
  };

  const assess = useCallback(async () => {
    setIsAssessing(true);
    setAssessError(null);
    setClinicalRiskError(null);
    try {
      const payload = toPayload();
      const clinicalPayload = toClinicalFeaturesPayload();

      // Run both in parallel; an XGBoost failure must not block the rule-engine result.
      const [assessResult, riskResult] = await Promise.allSettled([
        fetch(`${API_BASE}/blood/assess`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }),
        fetch(`${API_BASE}/clinical/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(clinicalPayload),
        }),
      ]);

      if (assessResult.status !== 'fulfilled' || !assessResult.value.ok) {
        const res = assessResult.status === 'fulfilled' ? assessResult.value : null;
        const e = res ? await res.json().catch(() => ({})) : {};
        throw new Error(e.detail || `Assessment failed${res ? ` (${res.status})` : ''}`);
      }
      const data = await assessResult.value.json();
      setAssessment(data);
      onAssessed?.(data);

      if (riskResult.status === 'fulfilled' && riskResult.value.ok) {
        const riskData = await riskResult.value.json();
        setClinicalRisk(riskData);
        onClinicalRisk?.(riskData);
      } else {
        setClinicalRiskError('Clinical risk model unavailable.');
      }
    } catch (err) {
      setAssessError(err.message);
    } finally {
      setIsAssessing(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markers, ocrClinicalFeatures, onAssessed, onClinicalRisk]);

  const isPdf = !!file && (file.type === 'application/pdf' || (fileName || '').toLowerCase().endsWith('.pdf'));
  const suspicious = assessment?.recommend_ct === true;
  const clean = assessment?.recommend_ct === false;

  return (
    <div className="space-y-4">
      {/* Skip straight to CT analysis (the CT step does not need lab data). */}
      {!file && (
        <div className="flex justify-end">
          <button
            onClick={() => onProceedToCT?.({})}
            className="flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-cyan-300 transition-colors"
          >
            Skip blood report — go straight to CT scan <ArrowRight size={13} />
          </button>
        </div>
      )}

      {/* ── Step 1: upload & preview the report ── */}
      {!file ? (
        <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}>
          <div
            onClick={() => inputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            className={`relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300
              ${isDragging ? 'drop-zone-active border-cyan-400' : 'border-white/[0.08] hover:border-cyan-500/30 hover:bg-cyan-500/[0.02]'}`}
          >
            <input ref={inputRef} type="file" accept=".pdf,image/*"
              onChange={(e) => selectFile(e.target.files?.[0])} className="hidden" />
            <div className="flex items-center gap-4 px-5 py-4">
              <div className={`flex items-center justify-center w-12 h-12 rounded-xl border transition-colors
                ${isDragging ? 'bg-cyan-500/15 border-cyan-500/25' : 'bg-white/[0.03] border-white/[0.06]'}`}>
                {isDragging ? <CloudUpload size={20} className="text-cyan-400" /> : <Upload size={20} className="text-slate-500" />}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] font-medium text-slate-300">
                  <span className="text-cyan-400">Upload blood report</span> or drag & drop
                </p>
                <p className="text-[11px] text-slate-500">PDF, PNG, JPG — liver tumour marker / hepatic panel</p>
              </div>
              <button className="px-4 py-2 rounded-lg text-[12px] font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 hover:bg-cyan-500/15 transition-all">Browse</button>
            </div>
          </div>
        </motion.div>
      ) : (
        <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-4">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 min-w-0">
              <FileText size={15} className="text-cyan-400 shrink-0" />
              <p className="text-[13px] font-medium text-slate-300 truncate">{fileName}</p>
            </div>
            <button onClick={clearFile}
              className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300 px-2 py-1 rounded-md hover:bg-white/5 shrink-0">
              <X size={13} /> Remove
            </button>
          </div>

          {/* Report preview */}
          <div className="rounded-xl overflow-hidden border border-white/[0.06] bg-black/40 flex items-center justify-center">
            {isPdf ? (
              <iframe title="Blood report preview" src={previewUrl}
                className="w-full" style={{ height: 460, background: '#fff' }} />
            ) : (
              <img src={previewUrl} alt="Blood report preview" className="max-h-[460px] w-auto object-contain" />
            )}
          </div>

          {/* Step 2: extract values */}
          {!markers && (
            <div className="mt-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <p className="text-[11px] text-slate-500">Check the report looks right, then extract the tumour-marker values.</p>
              <button onClick={handleExtract} disabled={isExtracting}
                className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold bg-cyan-500 text-navy-950 hover:bg-cyan-400 transition-all disabled:opacity-60">
                {isExtracting ? <Loader2 size={15} className="animate-spin" /> : <FlaskConical size={15} />}
                {isExtracting ? 'Reading report…' : 'Extract values'}
              </button>
            </div>
          )}
        </motion.div>
      )}

      {extractError && (
        <div className="flex items-start gap-2.5 p-4 rounded-xl bg-red-500/10 border border-red-500/25 text-red-300 text-[13px]">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <p className="leading-relaxed">{extractError}</p>
        </div>
      )}

      {/* ── Editable marker review ── */}
      <AnimatePresence>
        {markers && (
          <motion.div
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="glass-panel p-5"
          >
            <div className="flex items-center gap-2 mb-1">
              <FlaskConical size={15} className="text-cyan-400" />
              <h3 className="text-[13px] font-semibold text-slate-100">Review extracted values</h3>
            </div>
            <p className="text-[11px] text-slate-500 mb-4">
              OCR can misread digits — please verify against your report before assessing.
            </p>

            {noMarkers && (
              <div className="flex items-start gap-2.5 mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/25 text-amber-300 text-[12.5px]">
                <AlertTriangle size={15} className="mt-0.5 shrink-0" />
                <p className="leading-relaxed">
                  No tumour-marker values were recognised — this may not be a liver tumour marker /
                  hepatic panel report. Check the file, or enter the values manually below.
                </p>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {MARKER_FIELDS.map((f) => (
                <div key={f.key} className="rounded-xl bg-white/[0.02] border border-white/[0.06] px-3 py-2.5">
                  <label className="block text-[11px] text-slate-400 mb-1 truncate" title={f.label}>{f.label}</label>
                  <div className="flex items-center gap-2">
                    <input
                      value={markers[f.key]}
                      onChange={(e) => updateMarker(f.key, e.target.value)}
                      placeholder="—"
                      inputMode="decimal"
                      className="w-full bg-transparent text-[15px] font-semibold text-slate-100 outline-none placeholder:text-slate-600"
                    />
                    <span className="text-[10px] text-slate-500 font-mono whitespace-nowrap">{f.unit}</span>
                  </div>
                  <p className="text-[9.5px] text-slate-600 font-mono mt-0.5">ref {f.ref}</p>
                </div>
              ))}
            </div>

            {/* Raw OCR text (collapsible) */}
            {rawText && (
              <div className="mt-4">
                <button onClick={() => setShowRaw((s) => !s)}
                  className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300">
                  <ChevronDown size={13} className={`transition-transform ${showRaw ? 'rotate-180' : ''}`} />
                  {showRaw ? 'Hide' : 'Show'} raw OCR text
                </button>
                {showRaw && (
                  <pre className="mt-2 max-h-40 overflow-auto text-[10px] leading-relaxed text-slate-500 bg-black/30 rounded-lg p-3 whitespace-pre-wrap font-mono border border-white/[0.05]">{rawText}</pre>
                )}
              </div>
            )}

            <div className="mt-5 flex justify-end">
              <button
                onClick={assess}
                disabled={isAssessing}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/25 transition-all disabled:opacity-50"
              >
                {isAssessing ? <Loader2 size={15} className="animate-spin" /> : <FlaskConical size={15} />}
                Assess Risk
              </button>
            </div>

            {assessError && (
              <p className="mt-3 text-xs text-red-400/80">{assessError}</p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Assessment verdict ── */}
      <AnimatePresence>
        {assessment && (
          <motion.div
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            className={`rounded-2xl p-5 border ${suspicious
              ? 'bg-red-500/[0.1] border-red-500/30'
              : 'bg-emerald-500/[0.1] border-emerald-500/30'}`}
          >
            <div className="flex items-start gap-3">
              <div className={`mt-0.5 flex items-center justify-center w-9 h-9 rounded-xl shrink-0 ${suspicious ? 'bg-red-500/15 text-red-400' : 'bg-emerald-500/15 text-emerald-400'}`}>
                {suspicious ? <AlertTriangle size={18} /> : <CheckCircle2 size={18} />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className={`text-[15px] sm:text-[16px] font-bold ${suspicious ? 'text-red-200' : 'text-emerald-200'}`}>
                    {suspicious ? 'Tumour markers abnormal — CT scan recommended'
                      : clean ? 'Tumour markers within normal limits'
                      : 'Assessment complete'}
                  </h3>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded-md bg-white/[0.08] text-slate-300">
                    score {assessment.score} · {assessment.risk_level}
                  </span>
                </div>
                <p className="text-[13.5px] text-slate-200 mt-2.5 leading-relaxed">{assessment.interpretation}</p>

                {assessment.reasons?.length > 0 && (
                  <ul className="mt-3.5 space-y-1.5">
                    {assessment.reasons.map((r, i) => (
                      <li key={i} className="flex items-start gap-2.5 text-[13px] text-slate-200 leading-relaxed">
                        <span className="mt-[7px] w-1.5 h-1.5 rounded-full bg-red-400 shrink-0" />
                        <span>{r}</span>
                      </li>
                    ))}
                  </ul>
                )}

                {clinicalRisk && (
                  <div className="mt-4 p-3 rounded-xl bg-black/20 border border-white/[0.06]">
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <div className="flex items-center gap-2">
                        <Activity size={13} className="text-cyan-400" />
                        <span className="text-[11px] font-semibold text-slate-300">Clinical model (XGBoost)</span>
                        <span className={`text-[11px] font-mono px-1.5 py-0.5 rounded-md ${clinicalRisk.recommend_ct
                          ? 'bg-red-500/15 text-red-300' : 'bg-emerald-500/15 text-emerald-300'}`}>
                          {clinicalRisk.recommend_ct ? 'Elevated risk' : 'No significant risk'}
                        </span>
                      </div>
                      <button
                        onClick={() => setShowTreeModal(true)}
                        className="flex items-center gap-1.5 text-[10.5px] text-slate-500 hover:text-cyan-300 px-2 py-1 rounded-md hover:bg-white/5 transition-colors shrink-0"
                      >
                        <GitBranch size={12} /> View Decision Tree
                      </button>
                    </div>
                    <p className="text-[12px] text-slate-400 leading-relaxed">{clinicalRisk.interpretation}</p>
                  </div>
                )}
                {clinicalRiskError && (
                  <p className="mt-2 text-[11px] text-slate-500">{clinicalRiskError}</p>
                )}

                <div className="mt-4">
                  {suspicious ? (
                    <button
                      onClick={() => onProceedToCT?.(toPayload())}
                      className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold bg-cyan-500 text-navy-950 hover:bg-cyan-400 transition-all"
                    >
                      Proceed to CT Scan Analysis <ArrowRight size={15} />
                    </button>
                  ) : (
                    <div className="flex items-center gap-4">
                      <p className="text-[12px] text-emerald-400/80">
                        No cross-sectional imaging indicated. Routine surveillance advised.
                      </p>
                      <button
                        onClick={() => onProceedToCT?.(toPayload())}
                        className="text-[11px] text-slate-500 hover:text-slate-300 underline underline-offset-2"
                      >
                        Upload CT anyway
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <DecisionTreeModal open={showTreeModal} onClose={() => setShowTreeModal(false)} />
    </div>
  );
}
