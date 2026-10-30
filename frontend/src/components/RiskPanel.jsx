import { motion } from 'framer-motion';
import PanelWrapper from './PanelWrapper';
import { ShieldAlert, TrendingUp } from 'lucide-react';

export default function RiskPanel({ analysisResult, delay = 4 }) {
  const hasTumor = !!analysisResult?.has_tumor;
  const areaPercentage = analysisResult?.tumor_area_percentage || 0;

  // Severity comes from the segmentation result (tumor presence and extent).
  const severity = !hasTumor ? 'low' : areaPercentage >= 5 ? 'high' : 'medium';

  const severityColors = {
    low: { ring: '#10b981', text: 'text-emerald-400' },
    medium: { ring: '#f59e0b', text: 'text-amber-400' },
    high: { ring: '#ef4444', text: 'text-red-400' },
  };

  const sc = severityColors[severity];
  const circumference = 2 * Math.PI * 70;
  // The gauge saturates at 20% tumor coverage.
  const gaugeFraction = hasTumor ? Math.min(areaPercentage / 20, 1) : 0;
  const gaugeOffset = circumference * (1 - gaugeFraction);

  return (
    <PanelWrapper
      title="Risk Prediction"
      icon={ShieldAlert}
      badge={severity.toUpperCase()}
      badgeColor={severity === 'high' ? 'red' : severity === 'medium' ? 'amber' : 'green'}
      delay={delay}
    >
      {/* Tumor coverage gauge */}
      <div className="flex justify-center mb-4">
        <div className="relative w-44 h-44">
          <svg className="circular-progress w-full h-full" viewBox="0 0 160 160">
            {/* Background ring */}
            <circle cx="80" cy="80" r="70" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="8" />
            {/* Coverage arc */}
            <motion.circle
              cx="80" cy="80" r="70" fill="none"
              stroke={sc.ring}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset: gaugeOffset }}
              transition={{ duration: 1.8, delay: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
            />
            {/* Inner glow ring */}
            <circle cx="80" cy="80" r="62" fill="none" stroke="rgba(255,255,255,0.02)" strokeWidth="1" />
          </svg>

          {/* Center Content */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.span
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 1 }}
              className={`text-3xl font-bold font-mono ${sc.text}`}
            >
              {hasTumor ? `${areaPercentage}%` : '—'}
            </motion.span>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">
              {hasTumor ? 'Tumor Coverage' : 'No Tumor Found'}
            </span>
          </div>
        </div>
      </div>

      {/* Segmentation finding */}
      <div className="p-2.5 rounded-lg bg-black/20 border border-white/[0.04] mb-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] text-slate-500 font-mono uppercase">Segmentation Finding</span>
          <span className={`text-[11px] font-mono font-semibold ${hasTumor ? 'text-red-400' : 'text-emerald-400'}`}>
            {hasTumor ? 'Tumor detected' : 'Clear'}
          </span>
        </div>
        <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(areaPercentage, 100)}%` }}
            transition={{ duration: 1.2, delay: 1, ease: 'easeOut' }}
            className={`h-full rounded-full bg-gradient-to-r ${hasTumor ? 'from-red-600 to-red-400' : 'from-emerald-600 to-emerald-400'}`}
          />
        </div>
      </div>

      {/* Risk Factors */}
      <div className="p-2.5 rounded-lg bg-black/20 border border-white/[0.04]">
        <div className="flex items-center gap-1.5 mb-2">
          <TrendingUp size={11} className="text-slate-500" />
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">Analysis Summary</span>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-1 h-1 rounded-full bg-red-400/60" />
            <span className="flex-1 text-[10px] text-slate-400">Segmentation</span>
            <span className="text-[10px] font-mono font-semibold text-slate-300">{hasTumor ? 'Tumor found' : 'Clear'}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-1 h-1 rounded-full bg-red-400/60" />
            <span className="flex-1 text-[10px] text-slate-400">Tumor Coverage</span>
            <span className="text-[10px] font-mono font-semibold text-slate-300">{areaPercentage}%</span>
          </div>
        </div>
      </div>
    </PanelWrapper>
  );
}
