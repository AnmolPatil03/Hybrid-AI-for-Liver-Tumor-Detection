import { useState, useCallback } from 'react';
import Landing from './components/Landing';
import App from './App';

/**
 * Shows the landing page first; the CTA opens the diagnostic workflow,
 * which starts with the blood report upload.
 */
export default function Root() {
  const [entered, setEntered] = useState(false);

  const handleEnter = useCallback(() => {
    window.scrollTo(0, 0);
    setEntered(true);
  }, []);

  const handleHome = useCallback(() => {
    window.scrollTo(0, 0);
    setEntered(false);
  }, []);

  return entered ? <App onHome={handleHome} /> : <Landing onEnter={handleEnter} />;
}
