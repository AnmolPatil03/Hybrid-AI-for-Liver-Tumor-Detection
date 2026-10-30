import { motion, AnimatePresence } from 'framer-motion';
import { Maximize2, Minimize2 } from 'lucide-react';
import { useState, useEffect } from 'react';

export default function PanelWrapper({ title, icon: Icon, badge, badgeColor = 'cyan', children, className = '', delay = 0 }) {
  const [expanded, setExpanded] = useState(false);

  const badgeColors = {
    cyan: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/20',
    red: 'bg-red-500/15 text-red-400 border-red-500/20',
    green: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
    amber: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
    purple: 'bg-purple-500/15 text-purple-400 border-purple-500/20',
  };

  // While expanded: close on Escape and lock background scroll.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e) => { if (e.key === 'Escape') setExpanded(false); };
    window.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [expanded]);

  const Header = ({ isExpanded }) => (
    <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06] shrink-0">
      <div className="flex items-center gap-2.5">
        {Icon && (
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/15">
            <Icon size={14} className="text-cyan-400" />
          </div>
        )}
        <h3 className="text-[13px] font-semibold text-slate-200 tracking-wide uppercase">
          {title}
        </h3>
      </div>
      <div className="flex items-center gap-2">
        {badge && (
          <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border ${badgeColors[badgeColor]}`}>
            {badge}
          </span>
        )}
        <button
          onClick={() => setExpanded(!isExpanded)}
          className="p-1.5 rounded-md hover:bg-white/5 text-slate-500 hover:text-slate-300 transition-colors"
          data-tooltip={isExpanded ? 'Exit fullscreen' : 'Fullscreen'}
        >
          {isExpanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* In-grid card. When expanded, its body is emptied and `children` moves
          to the modal — rendering them in only one place at a time keeps any
          refs inside `children` (e.g. the measurement bounding box) from
          colliding across two mounted copies. The grid reflow is hidden behind
          the blurred backdrop. */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: delay * 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={`glass-panel glow-border-cyan flex flex-col overflow-hidden relative ${className}`}
      >
        <Header isExpanded={false} />
        <div className="flex-1 overflow-auto p-4">{!expanded && children}</div>
      </motion.div>

      {/* Expanded modal — pops to the front, blurs & dims everything else */}
      <AnimatePresence>
        {expanded && (
          <>
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              onClick={() => setExpanded(false)}
              className="fixed inset-0 z-[90] bg-navy-950/70 backdrop-blur-md"
            />
            <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6 md:p-10 pointer-events-none">
              <motion.div
                key="modal"
                initial={{ opacity: 0, scale: 0.96, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.97, y: 10 }}
                transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
                className={`glass-panel glow-border-cyan flex flex-col overflow-hidden relative
                  w-full max-w-[1200px] max-h-full pointer-events-auto
                  shadow-2xl shadow-black/60 ring-1 ring-white/10 ${className}`}
              >
                <Header isExpanded={true} />
                <div className="flex-1 overflow-auto p-4 sm:p-6">{children}</div>
              </motion.div>
            </div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
