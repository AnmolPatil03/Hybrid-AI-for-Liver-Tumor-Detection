import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FileText, Sparkles, Loader2, Copy, Check, FileDown, RefreshCw, AlertTriangle,
} from 'lucide-react';
import { API_BASE, MARKER_FIELDS } from '../config';
import { deriveMeasurements } from '../utils/measurements';

/**
 * Sends the pipeline results to POST /report/generate and renders the report.
 * "Download PDF" opens a printable A4 page with the CT scan, segmentation,
 * measurements, markers and report sections.
 *
 * Props:
 *   uploadedImage, patient, clinical, assessment, analysis, clinicalRisk
 */

// Base64 image blobs carry no textual meaning and bloat the request — drop them.
const IMAGE_KEYS = ['gradcam_image', 'segmentation_overlay', 'segmentation_mask'];

// on-screen rendering helpers
function inline(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={i} className="text-slate-200 font-semibold">{part.slice(2, -2)}</strong>
      : <span key={i}>{part}</span>
  );
}

function SectionBody({ body }) {
  const lines = (body || '').split('\n');
  const blocks = [];
  let bullets = [];
  const flush = () => { if (bullets.length) { blocks.push({ type: 'ul', items: bullets }); bullets = []; } };
  lines.forEach((line) => {
    const t = line.trim();
    if (t.startsWith('- ')) bullets.push(t.slice(2));
    else { flush(); if (t) blocks.push({ type: 'p', text: t }); }
  });
  flush();
  return (
    <div className="space-y-2">
      {blocks.map((b, i) =>
        b.type === 'ul' ? (
          <ul key={i} className="space-y-1">
            {b.items.map((it, j) => (
              <li key={j} className="flex gap-2 text-[12.5px] text-slate-400 leading-relaxed">
                <span className="mt-[7px] w-1 h-1 rounded-full bg-cyan-400/60 shrink-0" />
                <span>{inline(it)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p key={i} className="text-[12.5px] text-slate-400 leading-relaxed">{inline(b.text)}</p>
        )
      )}
    </div>
  );
}

// printable-HTML helpers
function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function bodyToHtml(body) {
  const lines = String(body || '').split('\n');
  const bold = (t) => escapeHtml(t).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  let html = '';
  let inList = false;
  lines.forEach((line) => {
    const t = line.trim();
    if (t.startsWith('- ')) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${bold(t.slice(2))}</li>`;
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      if (t) html += `<p>${bold(t)}</p>`;
    }
  });
  if (inList) html += '</ul>';
  return html;
}

export default function FinalReportPanel({ uploadedImage, patient, clinical, assessment, analysis, clinicalRisk }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const measurements = deriveMeasurements(analysis);

  const generate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cleanAnalysis = { ...(analysis || {}) };
      IMAGE_KEYS.forEach((k) => delete cleanAnalysis[k]);
      const res = await fetch(`${API_BASE}/report/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient: patient || null,
          clinical: clinical || null,
          assessment: assessment || null,
          analysis: cleanAnalysis,
          clinical_risk: clinicalRisk || null,
        }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Report generation failed (${res.status})`);
      }
      setReport(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [patient, clinical, assessment, analysis, clinicalRisk]);

  const toPlainText = useCallback(() => {
    if (!report) return '';
    let out = 'LIVER TUMOUR DIAGNOSTIC REPORT\n\n';
    if (report.summary) out += `Impression: ${report.summary}\n\n`;
    report.sections.forEach((s) => { out += `${s.title}\n${'-'.repeat(s.title.length)}\n${s.body}\n\n`; });
    out += `\n${report.disclaimer}`;
    return out;
  }, [report]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(toPlainText());
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch { /* clipboard unavailable */ }
  }, [toPlainText]);

  // ── build the printable report and open the print dialog (Save as PDF) ──────
  const downloadPdf = useCallback(() => {
    const ctImg = uploadedImage || null;
    const segImg = analysis?.segmentation_overlay ? `data:image/png;base64,${analysis.segmentation_overlay}` : null;
    const intensity = analysis?.tumor_intensity || null;
    const p = patient || {};
    const ageSex = p.age != null && p.sex ? `${p.age} / ${p.sex}` : (p.age != null ? String(p.age) : '—');

    // Images block (original CT + tumour segmentation)
    const figures = [];
    if (ctImg) figures.push(`<figure><img src="${ctImg}" alt="Original CT"/><figcaption>Original CT Scan</figcaption></figure>`);
    if (segImg) figures.push(`<figure><img src="${segImg}" alt="Tumour segmentation"/><figcaption>Tumour Segmentation</figcaption></figure>`);
    const imagesHtml = figures.length
      ? `<div class="imgs">${figures.join('')}</div>`
      : '<p class="muted">No CT imaging available.</p>';

    // Tumour size / measurements table
    const sizeHtml = measurements ? `
      <table>
        <tr><th>Width</th><td>${measurements.widthMm.toFixed(1)} mm</td><th>Height</th><td>${measurements.heightMm.toFixed(1)} mm</td></tr>
        <tr><th>Area</th><td>${Math.round(measurements.areaMm2).toLocaleString()} mm²</td><th>Volume</th><td>${Math.round(measurements.volumeMm3).toLocaleString()} mm³</td></tr>
        <tr><th>Perimeter</th><td>${measurements.perimeterMm.toFixed(1)} mm</td><th>Circularity</th><td>${measurements.circularity.toFixed(2)}</td></tr>
        ${measurements.areaPct != null ? `<tr><th>Coverage</th><td colspan="3">${measurements.areaPct.toFixed(2)}% of image</td></tr>` : ''}
        ${intensity ? `<tr><th>Intensity (0–255)</th><td colspan="3">min ${intensity.min} · mean ${intensity.mean} · max ${intensity.max} <span class="muted">(grayscale, not calibrated HU)</span></td></tr>` : ''}
      </table>
      ${measurements.spacingAssumed ? '<p class="muted" style="font-size:10px;">mm values assume ~0.7 mm/px in-plane spacing &amp; 2.5 mm slice thickness — DICOM spacing not available.</p>' : ''}`
      : '<p class="muted">No tumour detected / measurements unavailable.</p>';

    // Clinical markers table
    const flags = assessment?.flags || {};
    const markerRows = MARKER_FIELDS
      .filter((f) => clinical && clinical[f.key] != null)
      .map((f) => {
        const flag = flags[f.key];
        const cls = flag === 'high' ? 'flag-high' : (flag === 'low' ? 'flag-low' : (flag === 'borderline' ? 'flag-bord' : ''));
        const tag = flag && flag !== 'normal' ? ` <span class="${cls}">(${flag})</span>` : '';
        return `<tr><td>${escapeHtml(f.label)}</td><td>${escapeHtml(String(clinical[f.key]))} ${escapeHtml(f.unit)}${tag}</td><td class="muted">${escapeHtml(f.ref)}</td></tr>`;
      }).join('');
    const clinicalHtml = markerRows
      ? `<table><tr><th>Marker</th><th>Value</th><th>Reference</th></tr>${markerRows}</table>
         ${assessment ? `<p class="muted">Blood risk: <strong>${escapeHtml(assessment.risk_level || '—')}</strong> (score ${assessment.score ?? '—'})</p>` : ''}`
      : '<p class="muted">No blood-marker values provided.</p>';

    // Clinical risk model
    const clinicalRiskHtml = clinicalRisk ? `
      <table>
        <tr><th>Risk level</th><td><strong>${escapeHtml(clinicalRisk.risk_level || '—')}</strong></td></tr>
        <tr><th>CT scan recommended</th><td>${clinicalRisk.recommend_ct ? 'Yes' : 'No'}</td></tr>
      </table>` : '';

    // Report sections
    const reportHtml = report ? `
      ${report.summary ? `<div class="impression"><span class="tag">IMPRESSION</span><p>${escapeHtml(report.summary)}</p></div>` : ''}
      ${report.sections.map((s) => `<div class="sec"><h3>${escapeHtml(s.title)}</h3>${bodyToHtml(s.body)}</div>`).join('')}
    ` : '';

    const dateStr = new Date().toLocaleString();

    const html = `<!doctype html><html><head><meta charset="utf-8"/>
      <title>Liver Tumour Report${p.patient_id ? ' — ' + escapeHtml(p.patient_id) : ''}</title>
      <style>
        @page { size: A4; margin: 15mm; }
        * { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; }
        body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1a1f2b; background: #e5e7eb; }
        /* On-screen A4 sheet: a real page-sized card centred on a grey desk */
        .sheet { width: 210mm; min-height: 297mm; margin: 12mm auto; background: #fff; padding: 15mm; box-shadow: 0 6px 28px rgba(0,0,0,.18); }
        h1 { font-size: 19px; color: #1f4fd1; margin: 0 0 2px; }
        h2 { font-size: 13px; color: #1f4fd1; border-bottom: 2px solid #1f4fd1; padding-bottom: 3px; margin: 18px 0 8px; text-transform: uppercase; letter-spacing: .04em; }
        h3 { font-size: 12.5px; margin: 10px 0 4px; color: #111827; }
        p, li, td, th { font-size: 12px; line-height: 1.5; }
        .top { display:flex; justify-content:space-between; align-items:flex-end; border-bottom:2px solid #1f4fd1; padding-bottom:8px; }
        .muted { color:#6b7280; }
        table { width:100%; border-collapse:collapse; margin:4px 0; break-inside:avoid; }
        td, th { border:1px solid #e3e8f0; padding:5px 8px; text-align:left; vertical-align:top; }
        th { background:#f5f7fb; font-weight:600; white-space:nowrap; }
        h2 { break-after:avoid; }
        .sec, .impression { break-inside:avoid; }
        .imgs { display:flex; gap:12px; margin:6px 0; break-inside:avoid; }
        .imgs figure { flex:1; margin:0; break-inside:avoid; }
        .imgs img { width:100%; max-height:300px; object-fit:contain; background:#000; border:1px solid #ccc; border-radius:6px; }
        figcaption { font-size:10.5px; color:#6b7280; text-align:center; margin-top:4px; }
        .impression { background:#eaf1fb; border:1px solid #cddaf5; border-radius:6px; padding:8px 10px; margin:6px 0; }
        .impression .tag { font-size:9px; color:#1f4fd1; font-weight:700; letter-spacing:.08em; }
        .impression p { margin:3px 0 0; }
        .sec ul { margin:4px 0; padding-left:18px; }
        .sec p { margin:4px 0; }
        .flag-high { color:#d12b2b; font-weight:700; }
        .flag-low { color:#c47a00; font-weight:700; }
        .flag-bord { color:#c47a00; }
        @media print {
          .noprint { display: none; }
          body { background: #fff; }
          /* Let @page own the margins so every printed page is consistent */
          .sheet { width: auto; min-height: 0; margin: 0; padding: 0; box-shadow: none; }
        }
      </style></head><body>
      <div class="sheet">
      <div class="top">
        <div>
          <h1>Liver Tumour Diagnostic Report</h1>
          <div class="muted" style="font-size:11px;">Clinical + CT imaging decision support · Generated ${escapeHtml(dateStr)}</div>
        </div>
      </div>

      <h2>Patient</h2>
      <table>
        <tr><th>Name</th><td>${escapeHtml(p.name || '—')}</td><th>Patient ID</th><td>${escapeHtml(p.patient_id || '—')}</td></tr>
        <tr><th>Age / Sex</th><td>${escapeHtml(ageSex)}</td><th>Study date</th><td>${escapeHtml(p.study_date || '—')}</td></tr>
      </table>

      <h2>Imaging — Original CT &amp; Tumour Segmentation</h2>
      ${imagesHtml}

      <h2>Tumour Size &amp; Measurements</h2>
      ${sizeHtml}

      <h2>Clinical (Blood) Markers</h2>
      ${clinicalHtml}

      ${clinicalRiskHtml ? `<h2>Clinical Risk Model (XGBoost)</h2>${clinicalRiskHtml}` : ''}

      ${reportHtml ? `<h2>Report</h2>${reportHtml}` : ''}
      </div>

      <script>window.onload=function(){setTimeout(function(){window.focus();window.print();},350);};</script>
      </body></html>`;

    const w = window.open('', '_blank');
    if (!w) { setError('Please allow pop-ups for this site to download the PDF.'); return; }
    w.document.open();
    w.document.write(html);
    w.document.close();
  }, [uploadedImage, patient, clinical, assessment, analysis, clinicalRisk, report, measurements]);

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass-panel-glow p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <FileText size={16} className="text-cyan-400" />
          <h3 className="text-[14px] font-bold text-slate-100">Final Report</h3>
        </div>

        {report && (
          <div className="flex items-center gap-1.5">
            <button onClick={handleCopy}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] bg-white/[0.03] border border-white/[0.06] text-slate-400 hover:text-cyan-300 hover:border-cyan-500/20 transition-all">
              {copied ? <Check size={13} /> : <Copy size={13} />}{copied ? 'Copied' : 'Copy'}
            </button>
            <button onClick={downloadPdf}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/25 transition-all">
              <FileDown size={13} />Download PDF
            </button>
            <button onClick={generate} disabled={loading}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] bg-white/[0.03] border border-white/[0.06] text-slate-400 hover:text-cyan-300 hover:border-cyan-500/20 transition-all disabled:opacity-50">
              <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />Regenerate
            </button>
          </div>
        )}
      </div>

      {/* Empty state — generate button */}
      {!report && !loading && (
        <div className="flex flex-col items-center justify-center text-center py-8 gap-4">
          <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
            <Sparkles size={22} className="text-cyan-400" />
          </div>
          <div>
            <p className="text-[13px] font-medium text-slate-200">Generate the final report</p>
            <p className="text-[11px] text-slate-500 mt-1 max-w-md">
              Combine the blood assessment, clinical risk model, and CT imaging into one
              clinician-readable report, then download it as a PDF with the
              CT scan, segmentation, and tumour size.
            </p>
          </div>
          <div className="flex flex-col items-center gap-2.5">
            <button onClick={generate}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold bg-cyan-500 text-navy-950 hover:bg-cyan-400 transition-all">
              <Sparkles size={15} /> Generate Final Report
            </button>
            <button onClick={downloadPdf}
              className="flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-cyan-300 transition-colors underline underline-offset-2">
              <FileDown size={12} /> Download PDF without report summary
            </button>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && !report && (
        <div className="flex items-center justify-center gap-3 py-10">
          <Loader2 size={16} className="text-cyan-400 animate-spin" />
          <p className="text-[13px] text-cyan-300 font-medium">Generating report…</p>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 mt-2">
          <AlertTriangle size={15} className="mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium">Something went wrong</p>
            <p className="text-xs mt-1 text-red-400/70">{error}</p>
          </div>
        </div>
      )}

      {/* Report */}
      <AnimatePresence>
        {report && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
            {report.summary && (
              <div className="rounded-xl px-4 py-3 bg-cyan-500/[0.06] border border-cyan-500/20">
                <p className="text-[10px] font-mono text-cyan-400/80 mb-1 tracking-wide">IMPRESSION</p>
                <p className="text-[13px] text-slate-200 leading-relaxed">{report.summary}</p>
              </div>
            )}

            {report.sections.map((s, i) => (
              <div key={i} className="rounded-xl px-4 py-3 bg-white/[0.02] border border-white/[0.06]">
                <h4 className="text-[12.5px] font-semibold text-slate-100 mb-2">{s.title}</h4>
                <SectionBody body={s.body} />
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
