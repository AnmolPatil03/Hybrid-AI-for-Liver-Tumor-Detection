import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import PanelWrapper from './PanelWrapper';
import { Scan, ZoomIn, ZoomOut, RotateCcw, Move } from 'lucide-react';
import { generateCTScan } from '../utils/imageGenerator';
import { defaultPatientInfo } from '../utils/uiConstants';

export default function OriginalScanPanel({ uploadedImage, patientInfo, delay = 0 }) {
  const [imageSrc, setImageSrc] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  useEffect(() => {
    if (uploadedImage) {
      setImageSrc(uploadedImage);
      setZoom(1);
      setPan({ x: 0, y: 0 });
    } else {
      setImageSrc(generateCTScan(512, 512));
    }
  }, [uploadedImage]);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 4));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5));
  const handleReset = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(prev => Math.min(Math.max(prev + delta, 0.5), 4));
  }, []);

  const handleMouseDown = (e) => {
    if (e.button === 0) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e) => {
    if (isPanning) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    }
  };

  const handleMouseUp = () => setIsPanning(false);

  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.addEventListener('wheel', handleWheel, { passive: false });
      return () => el.removeEventListener('wheel', handleWheel);
    }
  }, [handleWheel]);

  // Overlay demographics parsed from the uploaded report onto the defaults, so
  // the panel shows the same patient the report belongs to. Missing fields keep
  // their placeholder value.
  const info = patientInfo || {};
  const patient = {
    ...defaultPatientInfo,
    ...(info.name ? { name: info.name } : {}),
    ...(info.patient_id ? { id: info.patient_id } : {}),
    ...(info.age != null ? { age: info.age } : {}),
    ...(info.sex ? { sex: info.sex } : {}),
    ...(info.study_date ? { studyDate: info.study_date } : {}),
  };

  return (
    <PanelWrapper title="Original CT Scan" icon={Scan} badge="DICOM" badgeColor="cyan" delay={delay}>
      {/* Patient Info Bar */}
      <div className="mb-3 p-2.5 rounded-lg bg-black/30 border border-white/[0.04]">
        <div className="flex justify-between mb-1.5 pb-1.5 border-b border-white/[0.04]">
          <span className="text-[10px] text-slate-500 font-mono">Name</span>
          <span className="text-[10px] text-slate-200 font-mono font-medium">{patient.name}</span>
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <div className="flex justify-between">
            <span className="text-[10px] text-slate-500 font-mono">ID</span>
            <span className="text-[10px] text-cyan-400 font-mono font-medium">{patient.id}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[10px] text-slate-500 font-mono">Age/Sex</span>
            <span className="text-[10px] text-slate-300 font-mono">{patient.age}/{patient.sex}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[10px] text-slate-500 font-mono">Modality</span>
            <span className="text-[10px] text-slate-300 font-mono">{patient.modality}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[10px] text-slate-500 font-mono">Date</span>
            <span className="text-[10px] text-slate-300 font-mono">{patient.studyDate}</span>
          </div>
        </div>
        <div className="mt-1.5 pt-1.5 border-t border-white/[0.04]">
          <span className="text-[10px] text-slate-500 font-mono">Series: </span>
          <span className="text-[10px] text-slate-400 font-mono">{patient.series}</span>
        </div>
      </div>

      {/* Image Container */}
      <div
        ref={containerRef}
        className="dicom-viewport relative aspect-square cursor-grab active:cursor-grabbing select-none rounded-xl overflow-hidden border border-white/[0.06]"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Animated scan line */}
        <div className="absolute inset-0 z-20 pointer-events-none overflow-hidden">
          <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-cyan-500/15 to-transparent animate-scan" />
        </div>
        <AnimatePresence mode="wait">
          {imageSrc && (
            <motion.img
              key={imageSrc}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              src={imageSrc}
              alt="CT Scan"
              className="w-full h-full object-contain"
              style={{
                transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                transition: isPanning ? 'none' : 'transform 0.2s ease',
              }}
              draggable={false}
            />
          )}
        </AnimatePresence>

        {/* DICOM Corner Info */}
        <div className="absolute top-2 left-2 z-10">
          <p className="text-[9px] font-mono text-green-400/70">{patient.institution}</p>
          <p className="text-[9px] font-mono text-slate-500">{patient.series}</p>
        </div>
        <div className="absolute top-2 right-2 z-10">
          <p className="text-[9px] font-mono text-slate-500 text-right">W: 400 L: 60</p>
          <p className="text-[9px] font-mono text-slate-500 text-right">Slice: {patient.sliceThickness}</p>
        </div>
        <div className="absolute bottom-2 left-2 z-10">
          <p className="text-[9px] font-mono text-slate-500">Zoom: {(zoom * 100).toFixed(0)}%</p>
        </div>
      </div>

      {/* Zoom Controls */}
      <div className="flex items-center justify-center gap-1 mt-3">
        {[
          { icon: ZoomOut, action: handleZoomOut, tip: 'Zoom out' },
          { icon: RotateCcw, action: handleReset, tip: 'Reset' },
          { icon: ZoomIn, action: handleZoomIn, tip: 'Zoom in' },
        ].map(({ icon: BtnIcon, action, tip }) => (
          <button
            key={tip}
            onClick={action}
            data-tooltip={tip}
            className="p-2 rounded-lg bg-white/[0.03] border border-white/[0.06] hover:bg-cyan-500/10 hover:border-cyan-500/20 text-slate-500 hover:text-cyan-400 transition-all duration-200"
          >
            <BtnIcon size={14} />
          </button>
        ))}
      </div>
    </PanelWrapper>
  );
}
