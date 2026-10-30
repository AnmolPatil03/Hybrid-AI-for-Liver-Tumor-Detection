import { motion } from 'framer-motion';
import { processingStages } from '../utils/uiConstants';
import { Brain, CheckCircle2 } from 'lucide-react';

export default function ProcessingOverlay({ currentStage, progress }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-navy-950/90 backdrop-blur-xl"
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="glass-panel-glow p-8 rounded-2xl w-full max-w-md mx-4"
      >
        {/* Brain Icon */}
        <div className="flex justify-center mb-6">
          <motion.div
            animate={{ rotate: [0, 5, -5, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
            className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center"
          >
            <Brain size={28} className="text-cyan-400" />
          </motion.div>
        </div>

        <h3 className="text-center text-lg font-semibold text-slate-200 mb-1">AI Analysis in Progress</h3>
        <p className="text-center text-[12px] text-slate-500 mb-6">Running deep learning models on your scan</p>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
            <motion.div
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
              className="h-full rounded-full bg-gradient-to-r from-cyan-600 to-cyan-400"
              style={{ boxShadow: '0 0 15px rgba(34, 211, 238, 0.4)' }}
            />
          </div>
          <div className="flex justify-between mt-1.5">
            <span className="text-[10px] font-mono text-slate-500">Processing...</span>
            <span className="text-[10px] font-mono text-cyan-400">{Math.round(progress)}%</span>
          </div>
        </div>

        {/* Stages */}
        <div className="space-y-2">
          {processingStages.map((stage, i) => {
            const isDone = i < currentStage;
            const isCurrent = i === currentStage;
            return (
              <motion.div
                key={stage.label}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: i <= currentStage ? 1 : 0.3, x: 0 }}
                transition={{ duration: 0.3, delay: i * 0.05 }}
                className="flex items-center gap-2.5"
              >
                {isDone ? (
                  <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
                ) : isCurrent ? (
                  <div className="w-3.5 h-3.5 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin shrink-0" />
                ) : (
                  <div className="w-3.5 h-3.5 rounded-full border border-slate-700 shrink-0" />
                )}
                <span className={`text-[12px] ${isDone ? 'text-slate-400' : isCurrent ? 'text-cyan-400 font-medium' : 'text-slate-600'}`}>
                  {stage.label}
                </span>
              </motion.div>
            );
          })}
        </div>
      </motion.div>
    </motion.div>
  );
}
