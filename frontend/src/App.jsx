import { useState, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { processingStages } from './utils/uiConstants';
import { API_BASE } from './config';
import UploadArea from './components/UploadArea';
import OriginalScanPanel from './components/OriginalScanPanel';
import SegmentationPanel from './components/SegmentationPanel';
import HeatmapPanel from './components/HeatmapPanel';
import MeasurementsPanel from './components/MeasurementsPanel';
import RiskPanel from './components/RiskPanel';
import ProcessingOverlay from './components/ProcessingOverlay';
import WorkflowStepper from './components/WorkflowStepper';
import BloodReportStep from './components/BloodReportStep';
import FinalReportPanel from './components/FinalReportPanel';
import Navbar from './components/Navbar';
import { ArrowLeft, Sparkles } from 'lucide-react';

export default function App({ onHome }) {
  // Workflow: 'blood' (steps 1-2) → 'ct' (step 3).
  const [phase, setPhase] = useState('blood');
  const [assessment, setAssessment] = useState(null);
  const [clinicalValues, setClinicalValues] = useState({});
  const [clinicalRisk, setClinicalRisk] = useState(null);
  const [patientInfo, setPatientInfo] = useState(null);

  // CT / imaging analysis state
  const [uploadedImage, setUploadedImage] = useState(null);
  const [pendingFile, setPendingFile] = useState(null); // selected but not yet analyzed
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const [progress, setProgress] = useState(0);
  const [analysisReady, setAnalysisReady] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);

  const stepperCurrent = phase === 'ct' ? 3 : assessment ? 2 : 1;
  const ctUnlocked = phase === 'ct' || assessment?.recommend_ct === true;

  const handleProceedToCT = useCallback((values) => {
    setClinicalValues(values || {});
    setPhase('ct');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const backToBlood = useCallback(() => {
    setPhase('blood');
    setAnalysisReady(false);
    setAnalysisResult(null);
    setUploadedImage(null);
    setPendingFile(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  // files: { center, prev, next } for the 2.5D model.
  const runAnalysis = useCallback(async (files) => {
    setIsProcessing(true);
    setAnalysisReady(false);
    setAnalysisError(null);
    setCurrentStage(0);
    setProgress(0);

    // Simulated staged progress for UX while the API calls run.
    let stage = 0;
    const total = processingStages.reduce((a, s) => a + s.duration, 0);
    let elapsed = 0;
    const stageInterval = setInterval(() => {
      if (stage >= processingStages.length) { clearInterval(stageInterval); return; }
      setCurrentStage(stage);
      elapsed += processingStages[stage].duration;
      setProgress((elapsed / total) * 100);
      stage++;
    }, 600);

    try {
      // Segmentation + Seg-Grad-CAM
      const fd = new FormData();
      fd.append('file', files.center);
      fd.append('prev_file', files.prev);
      fd.append('next_file', files.next);
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: fd });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `API error: ${res.status}`);
      }
      setAnalysisResult(await res.json());
    } catch (err) {
      setAnalysisError(err.message);
      setAnalysisResult(null);
    } finally {
      clearInterval(stageInterval);
      setIsProcessing(false);
      setAnalysisReady(true);
      setProgress(100);
    }
  }, []);

  // Selecting files only shows a preview and clears old results.
  const handleImageUpload = useCallback((imageData, files) => {
    setUploadedImage(imageData);
    setPendingFile(files || null);
    setAnalysisReady(false);
    setAnalysisResult(null);
    setAnalysisError(null);
  }, []);

  // Run the analysis on the previewed scan.
  const handleAnalyze = useCallback(() => {
    if (pendingFile) runAnalysis(pendingFile);
  }, [pendingFile, runAnalysis]);

  return (
    <div className="min-h-screen bg-navy-950 radiology-grid">
      <AnimatePresence>
        {isProcessing && <ProcessingOverlay currentStage={currentStage} progress={progress} />}
      </AnimatePresence>

      {/* Top navigation */}
      <Navbar variant="app" onHome={onHome} />

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 pt-24 pb-8">
        <WorkflowStepper current={stepperCurrent} ctUnlocked={ctUnlocked} />

        {/* ── Step 1 & 2: Blood report ── */}
        {phase === 'blood' && (
          <BloodReportStep
            onAssessed={setAssessment}
            onPatientInfo={setPatientInfo}
            onClinicalRisk={setClinicalRisk}
            onProceedToCT={handleProceedToCT}
          />
        )}

        {/* ── Step 3: CT scan analysis ── */}
        {phase === 'ct' && (
          <div className="space-y-5">
            <button onClick={backToBlood}
              className="flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-slate-300 transition-colors">
              <ArrowLeft size={13} /> Back to blood report
            </button>

            <div className="mb-1">
              <UploadArea onImageUpload={handleImageUpload} isProcessing={isProcessing} />
            </div>

            {/* Preview before analysis — shows the uploaded scan and an Analyze button */}
            {uploadedImage && pendingFile && !isProcessing && !analysisReady && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35 }}
                className="rounded-2xl border border-white/[0.08] bg-navy-900/40 p-5"
              >
                <div className="flex flex-col items-center gap-5 text-center">
                  <img
                    src={uploadedImage}
                    alt="Uploaded CT scan preview"
                    className="w-full max-w-[440px] h-80 sm:h-[440px] object-contain rounded-xl border border-white/[0.06] bg-black/40"
                  />
                  <div className="max-w-xl">
                    <p className="text-[14px] font-semibold text-slate-100">Scan ready for analysis</p>
                    <p className="text-[12px] text-slate-500 mt-1.5 leading-relaxed">
                      Review the preview, then run the AI pipeline — segmentation
                      and Grad-CAM.
                    </p>
                    <button
                      onClick={handleAnalyze}
                      className="mt-4 inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold text-white
                        bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400
                        shadow-lg shadow-cyan-500/20 transition-all"
                    >
                      <Sparkles size={15} />
                      Analyze scan
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {analysisReady && analysisResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 auto-rows-auto">
                <OriginalScanPanel uploadedImage={uploadedImage} patientInfo={patientInfo} delay={0} />
                <SegmentationPanel isProcessing={isProcessing}
                  segmentationOverlay={analysisResult?.segmentation_overlay}
                  hasTumor={analysisResult?.has_tumor}
                  analysisResult={analysisResult} delay={1} />
                <HeatmapPanel gradcamImage={analysisResult?.gradcam_image} delay={2} />
                <MeasurementsPanel uploadedImage={uploadedImage} analysisResult={analysisResult} delay={3} />
                <RiskPanel analysisResult={analysisResult} delay={4} />
              </div>
            )}

            {analysisReady && analysisResult && (
              <FinalReportPanel
                uploadedImage={uploadedImage}
                patient={patientInfo}
                clinical={clinicalValues}
                assessment={assessment}
                analysis={analysisResult}
                clinicalRisk={clinicalRisk}
              />
            )}

            {analysisError && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
                <p className="font-medium">Analysis Failed</p>
                <p className="text-xs mt-1 text-red-400/70">{analysisError}</p>
              </div>
            )}
          </div>
        )}
      </main>

    </div>
  );
}
