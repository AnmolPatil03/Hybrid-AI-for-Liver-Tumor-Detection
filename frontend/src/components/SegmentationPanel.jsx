import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import PanelWrapper from './PanelWrapper';
import { Layers } from 'lucide-react';
import { generateSegmentationOverlay } from '../utils/imageGenerator';

export default function SegmentationPanel({ segmentationOverlay, hasTumor, analysisResult, delay = 1 }) {
  const [fallbackSrc, setFallbackSrc] = useState(null);

  const detected = analysisResult?.has_tumor ?? hasTumor ?? false;
  const tumorPixels = analysisResult?.tumor_area_pixels;
  const coveragePct = analysisResult?.tumor_area_percentage;

  // Generate fallback canvas image only if no API result
  useEffect(() => {
    if (!segmentationOverlay) {
      generateSegmentationOverlay(512, 512).then(setFallbackSrc);
    }
  }, [segmentationOverlay]);

  // Use API image if available, otherwise use generated fallback
  const imageSrc = segmentationOverlay
    ? `data:image/png;base64,${segmentationOverlay}`
    : fallbackSrc;

  return (
    <PanelWrapper
      title="Tumor Segmentation"
      icon={Layers}
      badge={detected ? 'DETECTED' : 'CLEAR'}
      badgeColor={detected ? 'red' : 'cyan'}
      delay={delay}
    >
      {/* Image Viewport */}
      <div className="dicom-viewport relative aspect-square mb-3 rounded-xl overflow-hidden border border-white/[0.06] bg-black">
        {/* Scan line animation */}
        <div className="absolute inset-0 z-20 pointer-events-none overflow-hidden">
          <motion.div
            animate={{ y: ['0%', '100%'] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
            className="w-full h-[2px] bg-gradient-to-r from-transparent via-red-500/20 to-transparent"
          />
        </div>

        {imageSrc && (
          <motion.img
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
            src={imageSrc}
            alt="Segmentation overlay"
            className="w-full h-full object-contain"
            draggable={false}
          />
        )}

        {/* DICOM-style corner markers */}
        <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-[9px] font-mono text-red-400/90">TUMOR MASK</span>
        </div>
        <div className="absolute top-2 right-2 z-10">
          <span className="text-[9px] font-mono text-slate-500">SEG v2.1</span>
        </div>
        <div className="absolute bottom-2 left-2 z-10">
          <span className="text-[9px] font-mono text-slate-600">512×512</span>
        </div>
        <div className="absolute bottom-2 right-2 z-10">
          <span className="text-[9px] font-mono text-slate-500">
            {detected ? 'MASK OVERLAY' : 'NO MASK'}
          </span>
        </div>
      </div>

      {/* Stats from /analyze */}
      <div className="space-y-2">
        <StatRow label="Tumor Detected" value={detected ? 'Yes' : 'No'} color={detected ? 'text-red-400' : 'text-emerald-400'} />
        <StatRow label="Tumor Pixels" value={tumorPixels != null ? tumorPixels.toLocaleString() : '—'} color="text-slate-300" />
        <StatRow label="Coverage" value={coveragePct != null ? `${coveragePct.toFixed(2)}%` : '—'} color="text-amber-400" />

        {/* Coverage bar */}
        <div className="mt-2 pt-2 border-t border-white/[0.04]">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-slate-500 font-mono">Coverage of image</span>
            <span className="text-[10px] text-amber-400 font-mono font-semibold">{coveragePct != null ? `${coveragePct.toFixed(2)}%` : '—'}</span>
          </div>
          <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(coveragePct || 0, 100)}%` }}
              transition={{ duration: 1.2, delay: 0.5, ease: 'easeOut' }}
              className="h-full rounded-full bg-gradient-to-r from-amber-600 to-amber-400"
            />
          </div>
        </div>
      </div>
    </PanelWrapper>
  );
}

function StatRow({ label, value, color = 'text-slate-300' }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[11px] text-slate-500">{label}</span>
      <span className={`text-[11px] font-mono font-semibold ${color}`}>{value}</span>
    </div>
  );
}
