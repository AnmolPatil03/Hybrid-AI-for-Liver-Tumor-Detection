// Canvas-based image generation utilities for medical imaging visualization

// Seeded random for consistent noise patterns
function seededRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// Gaussian blur approximation using multiple box passes
function boxBlurCanvas(ctx, canvas, radius) {
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = imgData.data;
  const w = canvas.width;
  const h = canvas.height;
  const temp = new Uint8ClampedArray(pixels.length);

  for (let pass = 0; pass < 2; pass++) {
    const src = pass === 0 ? pixels : temp;
    const dst = pass === 0 ? temp : pixels;
    // Horizontal pass
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0, a = 0, count = 0;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = Math.min(w - 1, Math.max(0, x + dx));
          const idx = (y * w + nx) * 4;
          r += src[idx]; g += src[idx + 1]; b += src[idx + 2]; a += src[idx + 3];
          count++;
        }
        const idx = (y * w + x) * 4;
        dst[idx] = r / count; dst[idx + 1] = g / count; dst[idx + 2] = b / count; dst[idx + 3] = a / count;
      }
    }
    // Vertical pass
    const src2 = dst;
    const dst2 = src;
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let r = 0, g = 0, b = 0, a = 0, count = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          const ny = Math.min(h - 1, Math.max(0, y + dy));
          const idx = (ny * w + x) * 4;
          r += src2[idx]; g += src2[idx + 1]; b += src2[idx + 2]; a += src2[idx + 3];
          count++;
        }
        const idx = (y * w + x) * 4;
        dst2[idx] = r / count; dst2[idx + 1] = g / count; dst2[idx + 2] = b / count; dst2[idx + 3] = a / count;
      }
    }
    if (pass === 0) pixels.set(dst2);
  }
  ctx.putImageData(imgData, 0, 0);
}

/**
 * Generate a realistic-looking liver CT scan placeholder on canvas
 */
export function generateCTScan(width = 512, height = 512) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const rng = seededRandom(42);

  // Pure black background (CT standard)
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, width, height);

  const cx = width / 2;
  const cy = height / 2;

  // Subcutaneous fat layer (bright on CT - fat is bright in soft tissue window)
  ctx.beginPath();
  ctx.ellipse(cx, cy, width * 0.43, height * 0.37, 0, 0, Math.PI * 2);
  const fatGrad = ctx.createRadialGradient(cx, cy, width * 0.35, cx, cy, width * 0.43);
  fatGrad.addColorStop(0, 'rgb(100, 100, 100)');
  fatGrad.addColorStop(0.5, 'rgb(150, 150, 150)');
  fatGrad.addColorStop(1, 'rgb(80, 80, 80)');
  ctx.fillStyle = fatGrad;
  ctx.fill();

  // Muscle layer (medium gray)
  ctx.beginPath();
  ctx.ellipse(cx, cy, width * 0.40, height * 0.34, 0, 0, Math.PI * 2);
  const muscleGrad = ctx.createRadialGradient(cx, cy, width * 0.30, cx, cy, width * 0.40);
  muscleGrad.addColorStop(0, 'rgb(85, 85, 85)');
  muscleGrad.addColorStop(1, 'rgb(120, 120, 120)');
  ctx.fillStyle = muscleGrad;
  ctx.fill();

  // Peritoneal cavity / internal organs background
  ctx.beginPath();
  ctx.ellipse(cx, cy, width * 0.36, height * 0.30, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(50, 50, 50)';
  ctx.fill();

  // Liver - large anatomically-shaped organ on right side
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(cx - 20, cy - 70);
  ctx.bezierCurveTo(cx + 40, cy - 110, cx + 140, cy - 95, cx + 160, cy - 40);
  ctx.bezierCurveTo(cx + 170, cy + 10, cx + 155, cy + 55, cx + 120, cy + 70);
  ctx.bezierCurveTo(cx + 80, cy + 85, cx + 30, cy + 75, cx - 10, cy + 55);
  ctx.bezierCurveTo(cx - 40, cy + 35, cx - 50, cy - 10, cx - 40, cy - 40);
  ctx.bezierCurveTo(cx - 35, cy - 55, cx - 25, cy - 65, cx - 20, cy - 70);
  ctx.closePath();
  ctx.clip();

  // Liver parenchyma base - portal venous phase enhancement (liver is bright ~120-160 HU)
  const liverGrad = ctx.createRadialGradient(cx + 60, cy - 10, 10, cx + 60, cy - 10, 140);
  liverGrad.addColorStop(0, 'rgb(165, 165, 165)');
  liverGrad.addColorStop(0.4, 'rgb(145, 145, 145)');
  liverGrad.addColorStop(0.7, 'rgb(130, 130, 130)');
  liverGrad.addColorStop(1, 'rgb(115, 115, 115)');
  ctx.fillStyle = liverGrad;
  ctx.fillRect(cx - 60, cy - 120, 250, 220);

  // Liver texture - fine granular noise (realistic parenchyma)
  for (let i = 0; i < 6000; i++) {
    const x = cx - 50 + rng() * 230;
    const y = cy - 110 + rng() * 210;
    const brightness = 110 + rng() * 60;
    const size = 0.5 + rng() * 1.5;
    ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
    ctx.fillRect(x, y, size, size);
  }

  // Hepatic vessels (hypodense linear structures)
  const vessels = [
    { x1: cx + 30, y1: cy - 50, x2: cx + 80, y2: cy + 20, w: 3 },
    { x1: cx + 60, y1: cy - 70, x2: cx + 100, y2: cy - 10, w: 2.5 },
    { x1: cx + 90, y1: cy - 30, x2: cx + 130, y2: cy + 40, w: 2 },
    { x1: cx + 20, y1: cy + 10, x2: cx + 70, y2: cy + 50, w: 2 },
    { x1: cx + 50, y1: cy - 40, x2: cx + 45, y2: cy + 30, w: 1.5 },
  ];
  vessels.forEach(v => {
    ctx.beginPath();
    ctx.moveTo(v.x1, v.y1);
    ctx.lineTo(v.x2, v.y2);
    ctx.strokeStyle = `rgba(170, 170, 170, 0.7)`;
    ctx.lineWidth = v.w;
    ctx.stroke();
  });

  // Portal vein (enhanced in portal venous phase)
  ctx.beginPath();
  ctx.ellipse(cx + 40, cy + 5, 6, 5, 0.3, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(180, 180, 180)';
  ctx.fill();

  ctx.restore();

  // Tumor lesion - hypodense in portal venous phase
  const tumorCx = cx + 70;
  const tumorCy = cy - 20;
  const tumorRx = 48;
  const tumorRy = 42;

  // Tumor with irregular margins
  ctx.save();
  ctx.beginPath();
  // Create slightly irregular tumor boundary
  for (let angle = 0; angle < Math.PI * 2; angle += 0.05) {
    const irregularity = 1 + (rng() - 0.5) * 0.15;
    const rx = tumorRx * irregularity;
    const ry = tumorRy * irregularity;
    const x = tumorCx + Math.cos(angle) * rx;
    const y = tumorCy + Math.sin(angle) * ry;
    if (angle === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.clip();

  // Tumor body - heterogeneous enhancement (tumor is HYPODENSE = darker than liver)
  const tumorGrad = ctx.createRadialGradient(tumorCx - 5, tumorCy - 3, 2, tumorCx, tumorCy, tumorRx);
  tumorGrad.addColorStop(0, 'rgb(70, 70, 70)');
  tumorGrad.addColorStop(0.3, 'rgb(80, 80, 80)');
  tumorGrad.addColorStop(0.6, 'rgb(65, 65, 65)');
  tumorGrad.addColorStop(0.85, 'rgb(90, 90, 90)');
  tumorGrad.addColorStop(1, 'rgb(100, 100, 100)');
  ctx.fillStyle = tumorGrad;
  ctx.fillRect(tumorCx - 55, tumorCy - 50, 110, 100);

  // Internal heterogeneity
  for (let i = 0; i < 800; i++) {
    const a = rng() * Math.PI * 2;
    const r = rng() * 40;
    const x = tumorCx + Math.cos(a) * r;
    const y = tumorCy + Math.sin(a) * r * 0.87;
    const b = 55 + rng() * 50;
    ctx.fillStyle = `rgb(${b}, ${b}, ${b})`;
    ctx.fillRect(x, y, 1.5 + rng(), 1.5 + rng());
  }

  // Necrotic center (darker region)
  ctx.beginPath();
  ctx.ellipse(tumorCx + 4, tumorCy + 3, 14, 11, 0.4, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(35, 35, 35)';
  ctx.fill();

  ctx.restore();

  // Rim enhancement around tumor (characteristic of malignancy)
  ctx.beginPath();
  ctx.ellipse(tumorCx, tumorCy, tumorRx + 2, tumorRy + 2, 0, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(170, 170, 170, 0.9)';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Spine / vertebral body (bright cortical bone, dark marrow)
  ctx.beginPath();
  ctx.ellipse(cx - 2, cy + 75, 20, 18, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(220, 220, 220)';
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(cx - 2, cy + 75, 14, 12, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(90, 90, 90)';
  ctx.fill();
  // Spinal canal
  ctx.beginPath();
  ctx.ellipse(cx - 2, cy + 75, 5, 5, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(30, 30, 30)';
  ctx.fill();
  // Transverse processes
  ctx.beginPath();
  ctx.ellipse(cx - 28, cy + 72, 12, 5, -0.3, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(190, 190, 190)';
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(cx + 24, cy + 72, 12, 5, 0.3, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(190, 190, 190)';
  ctx.fill();

  // Spleen (left side, homogeneous)
  ctx.beginPath();
  ctx.ellipse(cx - 125, cy - 15, 38, 28, -0.2, 0, Math.PI * 2);
  const spleenGrad = ctx.createRadialGradient(cx - 125, cy - 15, 5, cx - 125, cy - 15, 35);
  spleenGrad.addColorStop(0, 'rgb(135, 135, 135)');
  spleenGrad.addColorStop(1, 'rgb(115, 115, 115)');
  ctx.fillStyle = spleenGrad;
  ctx.fill();
  // Spleen texture
  for (let i = 0; i < 300; i++) {
    const a = rng() * Math.PI * 2;
    const r = rng() * 28;
    const x = cx - 125 + Math.cos(a) * r;
    const y = cy - 15 + Math.sin(a) * r * 0.7;
    const b = 105 + rng() * 40;
    ctx.fillStyle = `rgb(${b}, ${b}, ${b})`;
    ctx.fillRect(x, y, 1, 1);
  }

  // Aorta (bright with contrast)
  ctx.beginPath();
  ctx.arc(cx - 8, cy + 40, 11, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(200, 200, 200)';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx - 8, cy + 40, 8, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(170, 170, 170)';
  ctx.fill();

  // IVC (slightly less enhanced)
  ctx.beginPath();
  ctx.ellipse(cx + 18, cy + 35, 9, 11, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(150, 150, 150)';
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(cx + 18, cy + 35, 6, 8, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(110, 110, 110)';
  ctx.fill();

  // Kidneys (posterior)
  // Right kidney
  ctx.beginPath();
  ctx.ellipse(cx + 70, cy + 55, 20, 12, 0.2, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(120, 120, 120)';
  ctx.fill();
  // Left kidney  
  ctx.beginPath();
  ctx.ellipse(cx - 75, cy + 50, 20, 12, -0.2, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(120, 120, 120)';
  ctx.fill();

  // Stomach (left anterior, with air/fluid level)
  ctx.beginPath();
  ctx.ellipse(cx - 60, cy - 40, 30, 22, 0.1, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(20, 20, 20)';
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(cx - 60, cy - 30, 25, 10, 0.1, 0, Math.PI * 2);
  ctx.fillStyle = 'rgb(55, 55, 55)';
  ctx.fill();

  // Bowel loops (scattered small circles with air)
  const bowelPositions = [
    { x: cx - 30, y: cy + 30, r: 8 },
    { x: cx - 50, y: cy + 45, r: 7 },
    { x: cx - 15, y: cy + 50, r: 6 },
    { x: cx + 5, y: cy + 55, r: 5 },
  ];
  bowelPositions.forEach(b => {
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgb(18, 18, 18)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r + 2, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgb(100, 100, 100)';
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Global medical imaging noise
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 5) { // Only add noise to non-background
      const noise = (rng() - 0.5) * 8;
      data[i] = Math.max(0, Math.min(255, data[i] + noise));
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
    }
  }
  ctx.putImageData(imageData, 0, 0);

  return canvas.toDataURL('image/png');
}

/**
 * Generate tumor segmentation overlay with realistic contour
 */
export function generateSegmentationOverlay(width = 512, height = 512) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const baseImage = new Image();
  const baseData = generateCTScan(width, height);

  return new Promise((resolve) => {
    baseImage.onload = () => {
      ctx.drawImage(baseImage, 0, 0);

      const cx = width / 2;
      const cy = height / 2;
      const tumorCx = cx + 70;
      const tumorCy = cy - 20;
      const tumorRx = 48;
      const tumorRy = 42;
      const rng = seededRandom(42);

      // Generate irregular tumor boundary path
      function tumorPath(scale = 1) {
        ctx.beginPath();
        const points = [];
        for (let angle = 0; angle < Math.PI * 2; angle += 0.04) {
          const irregularity = 1 + (rng() - 0.5) * 0.12;
          const rx = tumorRx * scale * irregularity;
          const ry = tumorRy * scale * irregularity;
          points.push({
            x: tumorCx + Math.cos(angle) * rx,
            y: tumorCy + Math.sin(angle) * ry,
          });
        }
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          const prev = points[i - 1];
          const curr = points[i];
          const cpx = (prev.x + curr.x) / 2;
          const cpy = (prev.y + curr.y) / 2;
          ctx.quadraticCurveTo(prev.x, prev.y, cpx, cpy);
        }
        ctx.closePath();
      }

      // Outer glow (soft, large)
      tumorPath(1.6);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.12)';
      ctx.fill();

      // Mid glow
      tumorPath(1.3);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
      ctx.fill();

      // Segmentation mask fill (strong semi-transparent red)
      tumorPath(1.1);
      ctx.fillStyle = 'rgba(255, 40, 40, 0.55)';
      ctx.fill();

      // Primary boundary contour (very bright red, thick with strong glow)
      tumorPath(1.1);
      ctx.strokeStyle = 'rgba(255, 50, 50, 1.0)';
      ctx.lineWidth = 5;
      ctx.shadowColor = 'rgba(255, 0, 0, 1.0)';
      ctx.shadowBlur = 20;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Second pass of contour for extra brightness
      tumorPath(1.1);
      ctx.strokeStyle = 'rgba(255, 100, 100, 0.8)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Inner bright contour
      tumorPath(0.88);
      ctx.strokeStyle = 'rgba(255, 150, 150, 0.6)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Dashed outer margin
      tumorPath(1.4);
      ctx.strokeStyle = 'rgba(255, 80, 80, 0.6)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 5]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Center crosshair (cyan, bold, very visible)
      ctx.strokeStyle = 'rgba(34, 211, 238, 1.0)';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = 'rgba(34, 211, 238, 0.8)';
      ctx.shadowBlur = 8;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(tumorCx - 45, tumorCy);
      ctx.lineTo(tumorCx - 10, tumorCy);
      ctx.moveTo(tumorCx + 10, tumorCy);
      ctx.lineTo(tumorCx + 45, tumorCy);
      ctx.moveTo(tumorCx, tumorCy - 45);
      ctx.lineTo(tumorCx, tumorCy - 10);
      ctx.moveTo(tumorCx, tumorCy + 10);
      ctx.lineTo(tumorCx, tumorCy + 45);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.shadowBlur = 0;

      // Center dot (larger, glowing)
      ctx.beginPath();
      ctx.arc(tumorCx, tumorCy, 5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(34, 211, 238, 1.0)';
      ctx.shadowColor = 'rgba(34, 211, 238, 1.0)';
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Measurement annotation line (width) - below tumor, very visible
      const annotY = tumorCy + tumorRy + 30;
      ctx.strokeStyle = 'rgba(34, 211, 238, 1.0)';
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(tumorCx - tumorRx - 8, annotY);
      ctx.lineTo(tumorCx + tumorRx + 8, annotY);
      ctx.stroke();
      // End ticks
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(tumorCx - tumorRx - 8, annotY - 8);
      ctx.lineTo(tumorCx - tumorRx - 8, annotY + 8);
      ctx.moveTo(tumorCx + tumorRx + 8, annotY - 8);
      ctx.lineTo(tumorCx + tumorRx + 8, annotY + 8);
      ctx.stroke();
      // Width label (large bold)
      ctx.fillStyle = 'rgba(34, 211, 238, 1.0)';
      ctx.font = 'bold 16px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.shadowColor = 'rgba(34, 211, 238, 0.6)';
      ctx.shadowBlur = 6;
      ctx.fillText('42.3 mm', tumorCx, annotY + 22);
      ctx.shadowBlur = 0;

      // Height measurement annotation (left of tumor)
      const annotX = tumorCx - tumorRx - 30;
      ctx.strokeStyle = 'rgba(34, 211, 238, 1.0)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(annotX, tumorCy - tumorRy - 8);
      ctx.lineTo(annotX, tumorCy + tumorRy + 8);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(annotX - 8, tumorCy - tumorRy - 8);
      ctx.lineTo(annotX + 8, tumorCy - tumorRy - 8);
      ctx.moveTo(annotX - 8, tumorCy + tumorRy + 8);
      ctx.lineTo(annotX + 8, tumorCy + tumorRy + 8);
      ctx.stroke();
      // Height label (large bold with glow)
      ctx.save();
      ctx.shadowColor = 'rgba(34, 211, 238, 0.6)';
      ctx.shadowBlur = 6;
      ctx.translate(annotX - 18, tumorCy);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('38.7 mm', 0, 0);
      ctx.restore();

      resolve(canvas.toDataURL('image/png'));
    };
    baseImage.src = baseData;
  });
}

/**
 * Generate Grad-CAM heatmap overlay with smooth Gaussian blending
 */
export function generateHeatmapOverlay(width = 512, height = 512) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const baseImage = new Image();
  const baseData = generateCTScan(width, height);

  return new Promise((resolve) => {
    baseImage.onload = () => {
      ctx.drawImage(baseImage, 0, 0);

      const cx = width / 2;
      const cy = height / 2;
      const tumorCx = cx + 70;
      const tumorCy = cy - 20;

      // Create heatmap on separate canvas for blending
      const heatCanvas = document.createElement('canvas');
      heatCanvas.width = width;
      heatCanvas.height = height;
      const hctx = heatCanvas.getContext('2d');

      // Multiple Gaussian-like activation spots - larger and more intense
      const spots = [
        { x: tumorCx, y: tumorCy, r: 80, intensity: 0.95 },
        { x: tumorCx - 15, y: tumorCy + 10, r: 55, intensity: 0.75 },
        { x: tumorCx + 18, y: tumorCy - 12, r: 50, intensity: 0.65 },
        { x: tumorCx - 25, y: tumorCy - 8, r: 40, intensity: 0.5 },
        { x: tumorCx + 25, y: tumorCy + 15, r: 38, intensity: 0.45 },
        { x: tumorCx - 8, y: tumorCy + 25, r: 35, intensity: 0.35 },
        { x: tumorCx + 10, y: tumorCy - 30, r: 28, intensity: 0.3 },
        // Peripheral weak activations (along liver boundary)
        { x: cx + 30, y: cy - 5, r: 35, intensity: 0.12 },
        { x: cx + 110, y: cy + 25, r: 25, intensity: 0.08 },
      ];

      spots.forEach(spot => {
        const grad = hctx.createRadialGradient(spot.x, spot.y, 0, spot.x, spot.y, spot.r);
        grad.addColorStop(0, `rgba(255, 0, 0, ${spot.intensity})`);
        grad.addColorStop(0.2, `rgba(255, 60, 0, ${spot.intensity * 0.85})`);
        grad.addColorStop(0.4, `rgba(255, 130, 0, ${spot.intensity * 0.6})`);
        grad.addColorStop(0.6, `rgba(255, 200, 0, ${spot.intensity * 0.35})`);
        grad.addColorStop(0.8, `rgba(255, 240, 50, ${spot.intensity * 0.15})`);
        grad.addColorStop(1, 'rgba(255, 255, 100, 0)');
        hctx.fillStyle = grad;
        hctx.fillRect(0, 0, width, height);
      });

      // Apply Gaussian blur to the heatmap layer for smooth blending
      boxBlurCanvas(hctx, heatCanvas, 3);
      boxBlurCanvas(hctx, heatCanvas, 2);

      // Blend heatmap onto base image with stronger effect
      ctx.globalCompositeOperation = 'screen';
      ctx.drawImage(heatCanvas, 0, 0);
      ctx.globalCompositeOperation = 'source-over';

      // Add stronger additive overlay for warm glow
      ctx.globalAlpha = 0.3;
      ctx.drawImage(heatCanvas, 0, 0);
      ctx.globalAlpha = 1.0;

      // Additional direct overlay for maximum visibility
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = 0.15;
      ctx.drawImage(heatCanvas, 0, 0);
      ctx.globalAlpha = 1.0;
      ctx.globalCompositeOperation = 'source-over';

      resolve(canvas.toDataURL('image/png'));
    };
    baseImage.src = baseData;
  });
}

/**
 * Generate measurement overlay with bounding box and annotations
 */
export function generateMeasurementOverlay(width = 512, height = 512) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const baseImage = new Image();
  const baseData = generateCTScan(width, height);

  return new Promise((resolve) => {
    baseImage.onload = () => {
      ctx.drawImage(baseImage, 0, 0);

      const cx = width / 2;
      const cy = height / 2;
      const tumorCx = cx + 70;
      const tumorCy = cy - 20;
      const bx = tumorCx - 55;
      const by = tumorCy - 50;
      const bw = 110;
      const bh = 96;

      // Bounding box (dashed cyan)
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.8)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(bx, by, bw, bh);
      ctx.setLineDash([]);

      // Corner brackets
      const cornerLen = 14;
      ctx.strokeStyle = 'rgba(34, 211, 238, 1.0)';
      ctx.lineWidth = 3;
      // Top-left
      ctx.beginPath();
      ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
      ctx.stroke();
      // Top-right
      ctx.beginPath();
      ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen);
      ctx.stroke();
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh);
      ctx.stroke();
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen);
      ctx.stroke();

      // Width measurement line (below bounding box)
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.9)';
      ctx.lineWidth = 1.5;
      const my = by + bh + 16;
      ctx.beginPath();
      ctx.moveTo(bx, my); ctx.lineTo(bx + bw, my);
      ctx.stroke();
      // Arrowheads
      ctx.beginPath();
      ctx.moveTo(bx, my - 6); ctx.lineTo(bx, my + 6);
      ctx.moveTo(bx + bw, my - 6); ctx.lineTo(bx + bw, my + 6);
      ctx.stroke();
      // Width label
      ctx.fillStyle = '#22d3ee';
      ctx.font = 'bold 14px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('42.3 mm', bx + bw / 2, my + 18);

      // Height measurement line (left of bounding box)
      const mx = bx - 16;
      ctx.strokeStyle = 'rgba(34, 211, 238, 0.9)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(mx, by); ctx.lineTo(mx, by + bh);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(mx - 6, by); ctx.lineTo(mx + 6, by);
      ctx.moveTo(mx - 6, by + bh); ctx.lineTo(mx + 6, by + bh);
      ctx.stroke();
      // Height label
      ctx.save();
      ctx.translate(mx - 16, by + bh / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('38.7 mm', 0, 0);
      ctx.restore();

      // Area annotation (right side, larger text)
      ctx.fillStyle = 'rgba(255, 80, 80, 1.0)';
      ctx.font = 'bold 13px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('Area: 1286.4 mm²', bx + bw + 12, tumorCy - 8);
      ctx.fillText('Vol: 18.5 cm³', bx + bw + 12, tumorCy + 10);
      ctx.fillStyle = 'rgba(34, 211, 238, 0.9)';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText('Perimeter: 134.8 mm', bx + bw + 12, tumorCy + 28);

      resolve(canvas.toDataURL('image/png'));
    };
    baseImage.src = baseData;
  });
}
