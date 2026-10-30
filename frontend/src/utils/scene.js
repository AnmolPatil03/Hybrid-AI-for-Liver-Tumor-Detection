import * as THREE from 'three';

/**
 * initScene
 * ---------
 * Renders the ambient 3D background for the landing page: two slowly rotating
 * wireframe icosahedron "globes" plus a drifting particle field, with a subtle
 * mouse-parallax on the camera. The canvas is a fixed, full-viewport backdrop
 * (see #liver-canvas in styles.css) so it sits behind every section and the
 * page is never blank while scrolling.
 *
 * Returns a cleanup function that stops the render loop and frees resources.
 */
export function initScene() {
  const canvas = document.getElementById('liver-canvas');
  if (!canvas) return () => {};

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.set(0, 0, 5);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // Outer wireframe globe
  const globe = new THREE.Mesh(
    new THREE.IcosahedronGeometry(12, 2),
    new THREE.MeshBasicMaterial({ color: 0x3b82f6, wireframe: true, transparent: true, opacity: 0.07 })
  );
  scene.add(globe);

  // Inner wireframe globe
  const innerGlobe = new THREE.Mesh(
    new THREE.IcosahedronGeometry(8, 1),
    new THREE.MeshBasicMaterial({ color: 0x06b6d4, wireframe: true, transparent: true, opacity: 0.1 })
  );
  scene.add(innerGlobe);

  // Drifting particle field
  const pCount = 600;
  const pGeo = new THREE.BufferGeometry();
  const pPos = new Float32Array(pCount * 3);
  for (let i = 0; i < pCount; i++) {
    pPos[i * 3] = (Math.random() - 0.5) * 28;
    pPos[i * 3 + 1] = (Math.random() - 0.5) * 28;
    pPos[i * 3 + 2] = (Math.random() - 0.5) * 18;
  }
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  const particles = new THREE.Points(
    pGeo,
    new THREE.PointsMaterial({
      color: 0x6cc6ff,
      size: 0.035,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  scene.add(particles);

  // Mouse parallax target
  let targetX = 0;
  let targetY = 0;
  const onMouse = (e) => {
    targetX = (e.clientX / window.innerWidth - 0.5) * 0.6;
    targetY = (e.clientY / window.innerHeight - 0.5) * 0.6;
  };
  window.addEventListener('mousemove', onMouse);

  let rafId;
  const animate = () => {
    rafId = requestAnimationFrame(animate);

    globe.rotation.y += 0.0012;
    globe.rotation.x += 0.0005;
    innerGlobe.rotation.y -= 0.0016;
    innerGlobe.rotation.z += 0.0009;
    particles.rotation.y += 0.0003;
    particles.rotation.x += 0.0001;

    // Ease camera toward the mouse for a subtle depth effect
    camera.position.x += (targetX - camera.position.x) * 0.04;
    camera.position.y += (-targetY - camera.position.y) * 0.04;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
  };
  animate();

  const onResize = () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  };
  window.addEventListener('resize', onResize);

  return () => {
    cancelAnimationFrame(rafId);
    window.removeEventListener('resize', onResize);
    window.removeEventListener('mousemove', onMouse);
    pGeo.dispose();
    globe.geometry.dispose();
    globe.material.dispose();
    innerGlobe.geometry.dispose();
    innerGlobe.material.dispose();
    particles.material.dispose();
    renderer.dispose();
  };
}
