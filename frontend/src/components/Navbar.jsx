import { useEffect, useState } from 'react';

/**
 * Top navigation shared by the landing page and the dashboard.
 *
 *   variant="landing" (default) — transparent while pinned to the top of the
 *     hero, then solidifies (blur + border + shadow) on scroll. Section links
 *     smooth-scroll the page; the pill CTA launches the dashboard (onEnter).
 *   variant="app" — always solid. Drops the section links (they don't exist on
 *     the dashboard) and swaps the CTA for a Home action (onHome) that returns
 *     to the landing page. The brand also returns home.
 */

const LINKS = [
  { label: 'Overview', href: '#hero-section' },
  { label: 'Capabilities', href: '#capabilities' },
  { label: 'Workflow', href: '#workflow' },
  { label: 'Get Started', href: '#get-started' },
];

export default function Navbar({ onEnter, onHome, variant = 'landing' }) {
  const isApp = variant === 'app';
  // On the dashboard there is no hero, so the bar stays solid from the start.
  const [scrolled, setScrolled] = useState(isApp);

  useEffect(() => {
    if (isApp) return; // dashboard nav is always solid
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [isApp]);

  const scrollTop = () => window.scrollTo({ top: 0, behavior: 'smooth' });
  const onBrandClick = isApp ? onHome : scrollTop;

  return (
    <nav className={`landing-nav${scrolled ? ' scrolled' : ''}`}>
      {/* Brand */}
      <button type="button" className="nav-brand" onClick={onBrandClick}
        aria-label={isApp ? 'Hybrid AI — back to home' : 'Hybrid AI — back to top'}>
        <span className="nav-logo">
          <svg viewBox="0 0 24 24" fill="none" stroke="url(#navGrad)" strokeWidth="2.2"
            strokeLinecap="round" strokeLinejoin="round">
            <defs>
              <linearGradient id="navGrad" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
                <stop stopColor="#60a5fa" />
                <stop offset="0.5" stopColor="#a855f7" />
                <stop offset="1" stopColor="#22d3ee" />
              </linearGradient>
            </defs>
            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
          </svg>
        </span>
        <span className="nav-brand-text">
          <span className="nav-brand-name">Hybrid <span>AI</span></span>
          <span className="nav-brand-sub">Clinical + CT Diagnostics</span>
        </span>
      </button>

      {/* Links + action */}
      <div className="nav-right">
        {!isApp && (
          <div className="nav-links">
            {LINKS.map((l) => (
              <a key={l.href} href={l.href} className="nav-link">{l.label}</a>
            ))}
          </div>
        )}

        {isApp ? (
          <button type="button" className="nav-cta" onClick={onHome}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9.5 12 3l9 6.5M5 9v11h14V9" />
            </svg>
            Home
          </button>
        ) : (
          <button type="button" className="nav-cta" onClick={onEnter}>
            Launch Dashboard
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        )}
      </div>
    </nav>
  );
}
