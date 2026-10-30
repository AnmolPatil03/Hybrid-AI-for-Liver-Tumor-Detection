import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import PanelWrapper from './PanelWrapper';
import { Flame, Eye } from 'lucide-react';
import { generateHeatmapOverlay } from '../utils/imageGenerator';
import { heatmapInfo } from '../utils/uiConstants';

export default function HeatmapPanel({ gradcamImage, delay = 2 }) {
  const [fallbackSrc, setFallbackSrc] = useState(null);
  const heatmap = heatmapInfo;

  // Generate fallback canvas image only if no API result
  useEffect(() => {
    if (!gradcamImage) {
      generateHeatmapOverlay(512, 512).then(setFallbackSrc);
    }
  }, [gradcamImage]);

  // Use API image if available, otherwise use generated fallback
  const imageSrc = gradcamImage
    ? `data:image/png;base64,${gradcamImage}`
    : fallbackSrc;

  return (
    <PanelWrapper
      title="AI Attention Heatmap"
      icon={Flame}
      badge="Grad-CAM++"
      badgeColor="amber"
      delay={delay}
    >
      {/* Image Viewport — square frame so it zooms cleanly in the expanded modal */}
      <div className="dicom-viewport relative aspect-square mb-3 rounded-xl overflow-hidden border border-white/[0.06] bg-black">
        {/* Scan line animation */}
        <div className="absolute inset-0 z-20 pointer-events-none overflow-hidden">
          <motion.div
            animate={{ y: ['0%', '100%'] }}
            transition={{ duration: 5, repeat: Infinity, ease: 'linear' }}
            className="w-full h-[1px] bg-gradient-to-r from-transparent via-amber-500/15 to-transparent"
          />
        </div>

        {imageSrc && (
          <motion.img
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            src={imageSrc}
            alt="Grad-CAM heatmap"
            className="w-full h-full object-contain"
            draggable={false}
          />
        )}

        {/* Method tag */}
        <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5">
          <Eye size={10} className="text-amber-400" />
          <span className="text-[9px] font-mono text-amber-400/90">{heatmap.method}</span>
        </div>
        <div className="absolute top-2 right-2 z-10">
          <span className="text-[9px] font-mono text-slate-500">{heatmap.targetLayer}</span>
        </div>
      </div>

      {/* Color scale legend — below the image */}
      <div className="p-2 rounded-lg bg-black/40 border border-white/[0.06]">
        <div className="h-2.5 rounded-sm" style={{
          background: 'linear-gradient(90deg, rgba(0,0,180,0.8), rgba(0,180,0,0.8), rgba(255,255,0,0.8), rgba(255,140,0,0.8), rgba(255,0,0,0.9))'
        }} />
        <div className="flex justify-between mt-1.5">
          <span className="text-[9px] font-mono text-blue-300/80">Low attention</span>
          <span className="text-[9px] font-mono text-slate-400">Medium</span>
          <span className="text-[9px] font-mono text-red-300/80">High attention</span>
        </div>
      </div>

    </PanelWrapper>
  );
}
