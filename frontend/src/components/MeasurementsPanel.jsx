import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import PanelWrapper from './PanelWrapper';
import { Ruler, Box } from 'lucide-react';
import { deriveMeasurements } from '../utils/measurements';

export default function MeasurementsPanel({ uploadedImage, analysisResult, delay = 3 }) {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const roRef = useRef(null);
  const [bboxStyle, setBboxStyle] = useState(null);

  // Derive measurements from the API result (mm via assumed pixel spacing — see
  // utils/measurements.js). Null when there is no tumour to measure.
  const bbox = analysisResult?.tumor_bbox;
  const hasTumor = analysisResult?.has_tumor;
  const dm = deriveMeasurements(analysisResult);

  const m = dm ? {
    width: dm.widthMm.toFixed(1),
    height: dm.heightMm.toFixed(1),
    area: Math.round(dm.areaMm2),
    volume: Math.round(dm.volumeMm3),
    perimeter: dm.perimeterMm.toFixed(1),
    circularity: dm.circularity,
    spacingAssumed: dm.spacingAssumed,
  } : {
    width: '0.0', height: '0.0', area: 0, volume: 0, perimeter: '0.0', circularity: 0,
    spacingAssumed: false,
  };
  // Grayscale intensity (0-255) inside the mask, not Hounsfield Units.
  const intensity = analysisResult?.tumor_intensity || null;

  // Calculate the pixel position of the bounding box relative to the container.
  // The image uses object-contain, so we must account for the letterboxing offset.
  const recalcBbox = useCallback(() => {
    // Only truly hide the box when there is no tumour to draw.
    if (!bbox || !hasTumor) {
      setBboxStyle(null);
      return;
    }

    const img = imgRef.current;
    const container = containerRef.current;
    // Mid-transition (expanding into / collapsing out of the modal) the nodes
    // are briefly unmounted — keep the last box and recompute once they're back
    // rather than flashing it off.
    if (!img || !container) return;

    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    if (!natW || !natH) return; // image not decoded yet; onLoad/ref will retry

    // The rendered size of the <img> element (its CSS box)
    const elemW = img.clientWidth;
    const elemH = img.clientHeight;

    // object-contain: image is scaled to fit inside the element box while preserving aspect ratio
    const scaleX = elemW / natW;
    const scaleY = elemH / natH;
    const scale = Math.min(scaleX, scaleY);

    const renderedW = natW * scale;
    const renderedH = natH * scale;

    // Offset from the top-left of the <img> element to the top-left of the visible image
    const offsetX = (elemW - renderedW) / 2;
    const offsetY = (elemH - renderedH) / 2;

    // Also need offset from container to the img element
    const containerRect = container.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    const imgOffsetX = imgRect.left - containerRect.left;
    const imgOffsetY = imgRect.top - containerRect.top;

    // Convert bbox coordinates (in original image pixels) to container pixels
    const left = imgOffsetX + offsetX + (bbox.x_min / natW) * renderedW;
    const top = imgOffsetY + offsetY + (bbox.y_min / natH) * renderedH;
    const width = (bbox.width / natW) * renderedW;
    const height = (bbox.height / natH) * renderedH;

    setBboxStyle({ left, top, width, height });
  }, [bbox, hasTumor]);

  // Callback ref for the viewport: re-attach a ResizeObserver whenever the
  // container node changes (e.g. the panel remounts into the fullscreen modal),
  // so the bounding box is recomputed for the new size. A plain effect would
  // keep observing the old, detached node.
  const setContainerRef = useCallback((node) => {
    containerRef.current = node;
    if (roRef.current) { roRef.current.disconnect(); roRef.current = null; }
    if (node && typeof ResizeObserver !== 'undefined') {
      roRef.current = new ResizeObserver(() => recalcBbox());
      roRef.current.observe(node);
    }
    recalcBbox();
  }, [recalcBbox]);

  // Callback ref for the <img>: when it remounts (into/out of the modal) and is
  // already decoded (cached), `onLoad` may not fire — so recompute here after
  // layout settles.
  const setImgRef = useCallback((node) => {
    imgRef.current = node;
    if (node && node.complete && node.naturalWidth) {
      requestAnimationFrame(() => recalcBbox());
    }
  }, [recalcBbox]);

  useEffect(() => {
    recalcBbox();
    window.addEventListener('resize', recalcBbox);
    return () => {
      window.removeEventListener('resize', recalcBbox);
      if (roRef.current) { roRef.current.disconnect(); roRef.current = null; }
    };
  }, [recalcBbox]);

  const measurements = [
    { label: 'Width', value: `${m.width} mm`, icon: '↔' },
    { label: 'Height', value: `${m.height} mm`, icon: '↕' },
    { label: 'Area', value: `${Number(m.area).toLocaleString()} mm²`, icon: '▢' },
    { label: 'Volume', value: `${Number(m.volume).toLocaleString()} mm³`, icon: '◻' },
    { label: 'Perimeter', value: `${m.perimeter} mm`, icon: '○' },
    { label: 'Circularity', value: Number(m.circularity).toFixed(2), icon: '◎' },
  ];

  return (
    <PanelWrapper
      title="Tumor Measurements"
      icon={Ruler}
      badge={`${m.width}×${m.height}mm`}
      badgeColor="cyan"
      delay={delay}
    >
      {/* Image with bounding box */}
      <div
        ref={setContainerRef}
        className="dicom-viewport relative aspect-[16/10] mb-3 rounded-xl overflow-hidden border border-white/[0.06] bg-black/50"
      >
        {uploadedImage && (
          <motion.img
            ref={setImgRef}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            src={uploadedImage}
            alt="Original scan"
            onLoad={recalcBbox}
            className="w-full h-full object-contain"
            draggable={false}
          />
        )}

        {/* Bounding box overlay — positioned with computed pixel values */}
        {bboxStyle && (
          <div
            className="absolute border-2 border-dashed border-red-500 pointer-events-none"
            style={{
              left: `${bboxStyle.left}px`,
              top: `${bboxStyle.top}px`,
              width: `${bboxStyle.width}px`,
              height: `${bboxStyle.height}px`,
              boxShadow: '0 0 10px rgba(239,68,68,0.5), inset 0 0 10px rgba(239,68,68,0.08)',
            }}
          >
            {/* Corner brackets */}
            <div className="absolute -top-[3px] -left-[3px] w-3 h-3 border-t-2 border-l-2 border-red-500" />
            <div className="absolute -top-[3px] -right-[3px] w-3 h-3 border-t-2 border-r-2 border-red-500" />
            <div className="absolute -bottom-[3px] -left-[3px] w-3 h-3 border-b-2 border-l-2 border-red-500" />
            <div className="absolute -bottom-[3px] -right-[3px] w-3 h-3 border-b-2 border-r-2 border-red-500" />

            {/* Width label (bottom) */}
            <div className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[10px] font-mono text-red-400 font-bold whitespace-nowrap bg-black/60 px-1.5 py-0.5 rounded border border-red-500/20">
              {m.width} mm
            </div>
            {/* Height label (left) */}
            <div className="absolute top-1/2 -left-1 -translate-x-full -translate-y-1/2 text-[10px] font-mono text-red-400 font-bold whitespace-nowrap bg-black/60 px-1.5 py-0.5 rounded border border-red-500/20">
              {m.height} mm
            </div>
          </div>
        )}

        <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5 bg-black/60 px-2 py-1 rounded border border-red-500/15">
          <Box size={10} className="text-red-400" />
          <span className="text-[9px] font-mono text-red-400">BOUNDING BOX</span>
        </div>
      </div>

      {/* Measurement Grid */}
      {m.spacingAssumed && (
        <p className="text-[9.5px] text-slate-600 mb-2 leading-relaxed">
          mm values assume {`~0.7 mm/px`} spacing &amp; 2.5 mm slice — DICOM spacing not available.
        </p>
      )}
      <div className="grid grid-cols-2 gap-2 mb-3">
        {measurements.map((item, i) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.1 * i + 0.5 }}
            className="p-2.5 rounded-lg bg-black/20 border border-white/[0.04] hover:border-cyan-500/15 transition-colors"
          >
            <div className="flex items-center gap-1.5 mb-1">
              <span className="text-[10px]">{item.icon}</span>
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">{item.label}</span>
            </div>
            <p className="text-[13px] font-mono font-semibold text-cyan-400">{item.value}</p>
          </motion.div>
        ))}
      </div>

      {/* Tumor intensity (0-255 grayscale, not HU) */}
      <div className="p-2.5 rounded-lg bg-black/20 border border-white/[0.04]">
        <div className="flex items-center justify-between mb-2">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Tumor Intensity (0–255)</p>
          <span className="text-[8.5px] font-mono px-1.5 py-0.5 rounded bg-white/[0.04] text-slate-500 border border-white/[0.06]">grayscale · not HU</span>
        </div>
        {intensity ? (
          <div className="flex items-center gap-3">
            <HUBar label="Min" value={intensity.min} max={255} color="from-blue-600 to-blue-400" />
            <HUBar label="Mean" value={intensity.mean} max={255} color="from-cyan-600 to-cyan-400" />
            <HUBar label="Max" value={intensity.max} max={255} color="from-amber-600 to-amber-400" />
          </div>
        ) : (
          <p className="text-[11px] text-slate-600">Not available (no tumour region detected).</p>
        )}
      </div>
    </PanelWrapper>
  );
}

function HUBar({ label, value, max, color }) {
  return (
    <div className="flex-1">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[9px] text-slate-500 font-mono">{label}</span>
        <span className="text-[9px] text-slate-300 font-mono font-semibold">{value}</span>
      </div>
      <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${(value / max) * 100}%` }}
          transition={{ duration: 1, delay: 0.8, ease: 'easeOut' }}
          className={`h-full rounded-full bg-gradient-to-r ${color}`}
        />
      </div>
    </div>
  );
}
