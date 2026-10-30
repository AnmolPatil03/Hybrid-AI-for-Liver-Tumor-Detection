import { useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Upload, CloudUpload, FileImage, X } from 'lucide-react';

/**
 * Collects the previous, center and next CT slices for the 2.5D model.
 *
 * onImageUpload(previewDataUrl, { center, prev, next }) fires once all three
 * are selected (preview = center slice). onImageUpload(null) clears it.
 */
const SLOTS = [
  { key: 'prev', label: 'Previous slice' },
  { key: 'center', label: 'Center slice (to analyze)' },
  { key: 'next', label: 'Next slice' },
];

export default function UploadArea({ onImageUpload, isProcessing }) {
  const [files, setFiles] = useState({ prev: null, center: null, next: null });
  const [draggingSlot, setDraggingSlot] = useState(null);
  const inputRefs = { prev: useRef(null), center: useRef(null), next: useRef(null) };

  const emit = useCallback((next) => {
    const { prev, center, next: nextSlice } = next;
    if (!prev || !center || !nextSlice) {
      onImageUpload(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => onImageUpload(e.target.result, { center, prev, next: nextSlice });
    reader.readAsDataURL(center);
  }, [onImageUpload]);

  const setSlotFile = useCallback((key, file) => {
    setFiles((prevFiles) => {
      const updated = { ...prevFiles, [key]: file || null };
      emit(updated);
      return updated;
    });
  }, [emit]);

  const clearAll = (e) => {
    e.stopPropagation();
    setFiles({ prev: null, center: null, next: null });
    onImageUpload(null);
    Object.values(inputRefs).forEach((ref) => { if (ref.current) ref.current.value = ''; });
  };

  const anyFile = files.prev || files.center || files.next;

  return (
    <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="space-y-2">
      <div className="flex items-center justify-between px-1">
        <p className="text-[11px] text-slate-500">
          All three consecutive slices are required for accurate 2.5D segmentation.
        </p>
        {anyFile && !isProcessing && (
          <button onClick={clearAll} className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300">
            <X size={11} /> Clear all
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        {SLOTS.map(({ key, label }) => {
          const file = files[key];
          const isDragging = draggingSlot === key;
          return (
            <div
              key={key}
              onClick={() => !isProcessing && inputRefs[key].current?.click()}
              onDrop={(e) => {
                e.preventDefault();
                setDraggingSlot(null);
                if (!isProcessing) setSlotFile(key, e.dataTransfer.files[0]);
              }}
              onDragOver={(e) => { e.preventDefault(); if (!isProcessing) setDraggingSlot(key); }}
              onDragLeave={() => setDraggingSlot(null)}
              className={`relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 px-4 py-4
                ${isDragging ? 'drop-zone-active border-cyan-400' : 'border-white/[0.08] hover:border-cyan-500/30 hover:bg-cyan-500/[0.02]'}
                ${isProcessing ? 'pointer-events-none opacity-60' : ''}`}
            >
              <input ref={inputRefs[key]} type="file" accept="image/*,.dcm"
                onChange={(e) => setSlotFile(key, e.target.files[0])} className="hidden" />
              <div className="flex items-center gap-3">
                <div className={`flex items-center justify-center w-9 h-9 rounded-xl border shrink-0 transition-colors
                  ${isDragging ? 'bg-cyan-500/15 border-cyan-500/25' : 'bg-white/[0.03] border-white/[0.06]'}`}>
                  {isDragging ? <CloudUpload size={16} className="text-cyan-400" />
                    : <Upload size={16} className="text-slate-500" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[10.5px] text-slate-500">{label} <span className="text-red-400/80">*</span></p>
                  {file ? (
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <FileImage size={12} className="text-cyan-400 shrink-0" />
                      <p className="text-[11.5px] font-medium text-slate-300 truncate">{file.name}</p>
                      <button
                        onClick={(e) => { e.stopPropagation(); setSlotFile(key, null); }}
                        className="p-0.5 rounded hover:bg-white/5 text-slate-500 hover:text-slate-300 shrink-0"
                      ><X size={11} /></button>
                    </div>
                  ) : (
                    <p className="text-[11px] text-slate-400 mt-0.5">Browse or drag & drop</p>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
