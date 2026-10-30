import { motion } from 'framer-motion';
import { FlaskConical, ClipboardCheck, ScanLine, Check } from 'lucide-react';

/**
 * 3-step workflow indicator:
 *   1. Blood Report  2. Risk Assessment  3. CT Scan Analysis
 *
 * Props:
 *   current: 1 | 2 | 3   — the active step
 *   ctUnlocked: bool      — whether step 3 is reachable
 */
const STEPS = [
  { id: 1, label: 'Blood Report', sub: 'Upload & extract', Icon: FlaskConical },
  { id: 2, label: 'Risk Assessment', sub: 'Marker analysis', Icon: ClipboardCheck },
  { id: 3, label: 'CT Scan Analysis', sub: 'Segmentation & Grad-CAM', Icon: ScanLine },
];

export default function WorkflowStepper({ current, ctUnlocked }) {
  return (
    <div className="glass-panel px-5 py-4 mb-5">
      <div className="flex items-center">
        {STEPS.map((s, i) => {
          const isDone = current > s.id;
          const isActive = current === s.id;
          const isLocked = s.id === 3 && !ctUnlocked && current < 3;
          const { Icon } = s;

          return (
            <div key={s.id} className="flex items-center flex-1 last:flex-none">
              <div className="flex items-center gap-3">
                <motion.div
                  initial={false}
                  animate={{ scale: isActive ? 1.05 : 1 }}
                  className={`relative flex items-center justify-center w-10 h-10 rounded-xl border transition-colors duration-300
                    ${isDone ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
                      : isActive ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-400 animate-pulse-glow'
                      : isLocked ? 'bg-white/[0.02] border-white/[0.05] text-slate-600'
                      : 'bg-white/[0.03] border-white/[0.08] text-slate-500'}`}
                >
                  {isDone ? <Check size={18} /> : <Icon size={18} />}
                </motion.div>
                <div className="hidden sm:block">
                  <p className={`text-[12px] font-semibold tracking-tight transition-colors
                    ${isActive ? 'text-slate-100' : isDone ? 'text-emerald-300/90' : 'text-slate-500'}`}>
                    {s.label}
                  </p>
                  <p className="text-[10px] text-slate-500 font-mono">{s.sub}</p>
                </div>
              </div>

              {i < STEPS.length - 1 && (
                <div className="flex-1 mx-3 h-px relative">
                  <div className="absolute inset-0 bg-white/[0.06]" />
                  <motion.div
                    className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-400/60 to-cyan-400/60"
                    initial={false}
                    animate={{ width: current > s.id ? '100%' : '0%' }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
