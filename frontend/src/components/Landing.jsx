import { useEffect } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { initScene } from '../utils/scene';
import Navbar from './Navbar';

/**
 * Landing page: three.js background, parallax hero, Features and How It Works
 * sections revealed on scroll (framer-motion), and a CTA into the dashboard.
 */

// Reusable scroll-reveal wrapper — animates in once when scrolled into view.
function Reveal({ children, className, x = 0, y = 44, delay = 0 }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, x, y }}
      whileInView={{ opacity: 1, x: 0, y: 0 }}
      viewport={{ once: true, amount: 0.25 }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

export default function Landing({ onEnter }) {
  const { scrollY } = useScroll();

  // Hero parallax — fade + lift the text and body image as the user scrolls.
  const heroOpacity = useTransform(scrollY, [0, 480], [1, 0]);
  const heroY = useTransform(scrollY, [0, 480], [0, -70]);
  const imgOpacity = useTransform(scrollY, [0, 560], [1, 0]);
  const imgScale = useTransform(scrollY, [0, 560], [1, 1.12]);
  const indicatorOpacity = useTransform(scrollY, [0, 160], [1, 0]);

  useEffect(() => {
    const cleanup = initScene();
    return () => {
      if (typeof cleanup === 'function') cleanup();
    };
  }, []);

  return (
    <div className="landing-bg font-sans">
      {/* Fixed professional top navigation */}
      <Navbar onEnter={onEnter} />

      {/* Fixed 3D background canvas (behind everything) */}
      <canvas id="liver-canvas"></canvas>

      {/* Body image sits OUTSIDE .landing-content so its `screen` blend can
          reach the dark scene behind it (no isolating stacking context). */}
      <motion.img
        className="hero-body-image"
        src="/human_body.png"
        alt=""
        style={{ opacity: imgOpacity, scale: imgScale }}
      />

      <div className="landing-content">
        {/* ==================== HERO ==================== */}
        <section id="hero-section">
          <motion.div className="hero-overlay" style={{ opacity: heroOpacity, y: heroY }}>
            <motion.div
              className="hero-badge"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <span className="pulse-dot"></span>
              <span>Hybrid AI Diagnostics</span>
            </motion.div>

            <motion.h1
              className="hero-title"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.1 }}
            >
              Liver Tumor
              <br />
              Hybrid AI Detection
            </motion.h1>

            <motion.p
              className="hero-subtitle"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              A hybrid pipeline that combines blood-marker screening with CT
              imaging — tumor segmentation and explainable Grad-CAM —
              and generates a clinician-ready report.
            </motion.p>
          </motion.div>

          <motion.div className="scroll-indicator" style={{ opacity: indicatorOpacity }}>
            <span>SCROLL</span>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 5v14M19 12l-7 7-7-7" />
            </svg>
          </motion.div>
        </section>

        {/* ==================== FEATURES ==================== */}
        <section id="capabilities" className="features-section">
          <Reveal><p className="section-label">Capabilities</p></Reveal>
          <Reveal delay={0.05}><h2 className="section-heading">Powered by Specialized AI Models</h2></Reveal>
          <div className="features-grid">
            <Reveal className="feature-card" x={-60}>
              <div className="feature-icon" style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.2), rgba(99,102,241,0.2))' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="12 2 2 7 12 12 22 7 12 2" />
                  <polyline points="2 17 12 22 22 17" />
                  <polyline points="2 12 12 17 22 12" />
                </svg>
              </div>
              <h3>Clinical Risk Model</h3>
              <p>A gradient-boosted (XGBoost) model scores lab markers and flags when imaging is warranted — transparent and inspectable down to individual decision trees.</p>
            </Reveal>
            <Reveal className="feature-card" delay={0.1}>
              <div className="feature-icon" style={{ background: 'linear-gradient(135deg, rgba(168,85,247,0.2), rgba(236,72,153,0.2))' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#a855f7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
              </div>
              <h3>Explainable Imaging</h3>
              <p>ConvNeXt-Tiny U-Net segmentation with Seg-Grad-CAM heatmaps show exactly where the model sees a tumor.</p>
            </Reveal>
            <Reveal className="feature-card" x={60} delay={0.15}>
              <div className="feature-icon" style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.2), rgba(16,185,129,0.2))' }}>
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                </svg>
              </div>
              <h3>AI Clinical Report</h3>
              <p>Generates a structured, clinician-readable report — grounded only in the model outputs — with one-click PDF export.</p>
            </Reveal>
          </div>
        </section>

        <Reveal><div className="section-divider"></div></Reveal>

        {/* ==================== HOW IT WORKS ==================== */}
        <section id="workflow" className="steps-section">
          <Reveal><p className="section-label">Workflow</p></Reveal>
          <Reveal delay={0.05}><h2 className="section-heading">How It Works</h2></Reveal>
          <div className="steps-timeline">
            <Reveal className="step-item" x={-70}>
              <div className="step-number" style={{ background: 'linear-gradient(135deg, #3b82f6, #6366f1)', color: 'white' }}>1</div>
              <div className="step-content">
                <h3>Blood Report Screening</h3>
                <p>Upload a lab report — OCR reads the tumour markers and patient details, and a rule engine flags whether imaging is warranted.</p>
              </div>
            </Reveal>
            <Reveal className="step-item" x={70} delay={0.05}>
              <div className="step-number" style={{ background: 'linear-gradient(135deg, #a855f7, #ec4899)', color: 'white' }}>2</div>
              <div className="step-content">
                <h3>CT Scan Analysis</h3>
                <p>Upload a CT scan for tumour segmentation and Grad-CAM explainability.</p>
              </div>
            </Reveal>
            <Reveal className="step-item" x={-70} delay={0.1}>
              <div className="step-number" style={{ background: 'linear-gradient(135deg, #06b6d4, #10b981)', color: 'white' }}>3</div>
              <div className="step-content">
                <h3>Final Report &amp; Export</h3>
                <p>Get a clinician-readable report combining every result — with tumour measurements and one-click PDF export.</p>
              </div>
            </Reveal>
          </div>
        </section>

        <Reveal><div className="section-divider" style={{ marginBottom: '40px' }}></div></Reveal>

        {/* ==================== ENTER CTA ==================== */}
        <section id="get-started" style={{ textAlign: 'center', padding: '30px 24px 120px' }}>
          <Reveal><p className="section-label">Get Started</p></Reveal>
          <Reveal delay={0.05}>
            <h2 className="section-heading" style={{ marginBottom: '32px' }}>Ready to analyze a scan?</h2>
          </Reveal>
          <Reveal delay={0.1}>
            <button type="button" className="landing-cta" onClick={onEnter}>
              Launch Diagnostic Dashboard
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </button>
          </Reveal>
        </section>
      </div>
    </div>
  );
}
