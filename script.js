const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('output');
const ctx = canvasEl.getContext('2d');
const styleLabelEl = document.getElementById('styleLabel');

const STYLES = [
  { name: 'None', apply: null },
  { name: 'Anime', apply: applyAnime },
  { name: 'Sketch', apply: applySketch },
  { name: 'Ghibli', apply: applyGhibli },
  { name: 'Red', apply: applyRed },
  { name: 'Pixelate', apply: applyPixelate },
  { name: 'Black & White', apply: applyBlackWhite },
  { name: 'Thermal', apply: applyThermal },
  { name: 'Green Pixel', apply: applyGreenPixel },
  { name: 'Neon Outline', apply: applyNeonOutline },
  { name: 'Pink Grid', apply: applyPinkGrid },
];

let styleIndex = 0;

// per-hand pinch state, keyed by handedness label ('Left' | 'Right')
const pinchState = { Left: false, Right: false };

const PINCH_THRESHOLD = 0.06; // normalized landmark distance

function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function onResults(results) {
  const w = canvasEl.width = window.innerWidth;
  const h = canvasEl.height = window.innerHeight;

  const vw = results.image.width;
  const vh = results.image.height;

  // "cover" fit: scale video to fill viewport, cropping overflow, centered
  const scale = Math.max(w / vw, h / vh);
  const drawW = vw * scale;
  const drawH = vh * scale;
  const offX = (w - drawW) / 2;
  const offY = (h - drawH) / 2;

  ctx.save();
  ctx.clearRect(0, 0, w, h);

  // mirror draw base frame
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, w - offX - drawW, offY, drawW, drawH);
  ctx.restore();

  // maps normalized landmark coords (0..1 in source video space) to
  // mirrored viewport pixel space, matching the cover-fit draw above
  const toScreen = (nx, ny) => ({
    x: w - (offX + nx * drawW),
    y: offY + ny * drawH,
  });

  const hands = results.multiHandLandmarks || [];
  const handedness = results.multiHandedness || [];

  const points = {}; // { Left: {thumb, index}, Right: {thumb, index} }

  hands.forEach((landmarks, i) => {
    const label = handedness[i]?.label; // 'Left' or 'Right' (from camera's mirrored POV, matches user's actual hand after our mirror)
    if (!label) return;

    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    points[label] = { thumb: thumbTip, index: indexTip };

    const pinching = dist(thumbTip, indexTip) < PINCH_THRESHOLD;
    if (pinchState[label] && !pinching) {
      // transition pinched -> released: advance style
      styleIndex = (styleIndex + 1) % STYLES.length;
      styleLabelEl.textContent = `Style: ${STYLES[styleIndex].name}`;
    }
    pinchState[label] = pinching;
  });

  if (points.Left && points.Right) {
    // fixed order quad: rotating a hand swaps thumb/index screen position,
    // crossing the connecting lines into a bowtie shape naturally
    const quad = [
      toScreen(points.Left.thumb.x, points.Left.thumb.y),
      toScreen(points.Left.index.x, points.Left.index.y),
      toScreen(points.Right.index.x, points.Right.index.y),
      toScreen(points.Right.thumb.x, points.Right.thumb.y),
    ];

    const xs = quad.map(p => p.x);
    const ys = quad.map(p => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const boxW = x1 - x0, boxH = y1 - y0;

    if (boxW > 10 && boxH > 10) {
      const styleFn = STYLES[styleIndex].apply;
      if (styleFn) {
        const region = ctx.getImageData(x0, y0, boxW, boxH);
        styleFn(region);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(quad[0].x, quad[0].y);
        for (let i = 1; i < quad.length; i++) ctx.lineTo(quad[i].x, quad[i].y);
        ctx.closePath();
        ctx.clip();

        const off = document.createElement('canvas');
        off.width = boxW; off.height = boxH;
        off.getContext('2d').putImageData(region, 0, 0);
        ctx.drawImage(off, x0, y0);
        ctx.restore();
      }

      ctx.save();
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(quad[0].x, quad[0].y);
      for (let i = 1; i < quad.length; i++) ctx.lineTo(quad[i].x, quad[i].y);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }
  }
}

// --- style filters, operate in-place on ImageData ---

function applyAnime(imageData) {
  const d = imageData.data;
  const levels = 4;
  const step = 255 / (levels - 1);
  for (let i = 0; i < d.length; i += 4) {
    d[i] = Math.round(d[i] / step) * step;
    d[i + 1] = Math.round(d[i + 1] / step) * step;
    d[i + 2] = Math.round(d[i + 2] / step) * step;
    const avg = (d[i] + d[i + 1] + d[i + 2]) / 3;
    d[i] = clamp(avg + (d[i] - avg) * 1.6);
    d[i + 1] = clamp(avg + (d[i + 1] - avg) * 1.6);
    d[i + 2] = clamp(avg + (d[i + 2] - avg) * 1.6);
  }
}

function applySketch(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.3 * data[i] + 0.59 * data[i + 1] + 0.11 * data[i + 2];
  }
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const p = y * width + x;
      const gx = gray[p - 1] - gray[p + 1];
      const gy = gray[p - width] - gray[p + width];
      const edge = Math.sqrt(gx * gx + gy * gy);
      const v = clamp(255 - edge * 2);
      const idx = p * 4;
      data[idx] = data[idx + 1] = data[idx + 2] = v;
    }
  }
}

function applyGhibli(imageData) {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    d[i] = clamp(r * 1.08 + 15);
    d[i + 1] = clamp(g * 1.05 + 10);
    d[i + 2] = clamp(b * 0.9);
    const avg = (d[i] + d[i + 1] + d[i + 2]) / 3;
    d[i] = clamp(d[i] + (avg - d[i]) * 0.15);
    d[i + 1] = clamp(d[i + 1] + (avg - d[i + 1]) * 0.15);
    d[i + 2] = clamp(d[i + 2] + (avg - d[i + 2]) * 0.15);
  }
}

function applyRed(imageData) {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = 0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2];
    d[i] = clamp(gray * 1.4 + 40);
    d[i + 1] = clamp(gray * 0.15);
    d[i + 2] = clamp(gray * 0.15);
  }
}

function applyPixelate(imageData) {
  const { data, width, height } = imageData;
  const block = 14;
  for (let by = 0; by < height; by += block) {
    for (let bx = 0; bx < width; bx += block) {
      let r = 0, g = 0, b = 0, count = 0;
      for (let y = by; y < Math.min(by + block, height); y++) {
        for (let x = bx; x < Math.min(bx + block, width); x++) {
          const idx = (y * width + x) * 4;
          r += data[idx]; g += data[idx + 1]; b += data[idx + 2];
          count++;
        }
      }
      r = Math.round(r / count); g = Math.round(g / count); b = Math.round(b / count);
      for (let y = by; y < Math.min(by + block, height); y++) {
        for (let x = bx; x < Math.min(bx + block, width); x++) {
          const idx = (y * width + x) * 4;
          data[idx] = r; data[idx + 1] = g; data[idx + 2] = b;
        }
      }
    }
  }
}

function applyBlackWhite(imageData) {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = clamp((0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2] - 128) * 1.2 + 128);
    d[i] = d[i + 1] = d[i + 2] = gray;
  }
}

function applyThermal(imageData) {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = (0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2]) / 255;
    // black -> blue -> green -> yellow -> red heat ramp
    let r, g, b;
    if (gray < 0.25) {
      const t = gray / 0.25;
      r = 0; g = 0; b = clamp(t * 255);
    } else if (gray < 0.5) {
      const t = (gray - 0.25) / 0.25;
      r = 0; g = clamp(t * 255); b = clamp(255 - t * 255);
    } else if (gray < 0.75) {
      const t = (gray - 0.5) / 0.25;
      r = clamp(t * 255); g = 255; b = 0;
    } else {
      const t = (gray - 0.75) / 0.25;
      r = 255; g = clamp(255 - t * 255); b = 0;
    }
    d[i] = r; d[i + 1] = g; d[i + 2] = b;
  }
}

function applyGreenPixel(imageData) {
  const { data, width, height } = imageData;
  const block = 14;
  for (let by = 0; by < height; by += block) {
    for (let bx = 0; bx < width; bx += block) {
      let sum = 0, count = 0;
      for (let y = by; y < Math.min(by + block, height); y++) {
        for (let x = bx; x < Math.min(bx + block, width); x++) {
          const idx = (y * width + x) * 4;
          sum += 0.3 * data[idx] + 0.59 * data[idx + 1] + 0.11 * data[idx + 2];
          count++;
        }
      }
      const gray = sum / count;
      for (let y = by; y < Math.min(by + block, height); y++) {
        for (let x = bx; x < Math.min(bx + block, width); x++) {
          const idx = (y * width + x) * 4;
          data[idx] = clamp(gray * 0.2);
          data[idx + 1] = clamp(gray * 1.3 + 20);
          data[idx + 2] = clamp(gray * 0.2);
        }
      }
    }
  }
}

function applyNeonOutline(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.3 * data[i] + 0.59 * data[i + 1] + 0.11 * data[i + 2];
  }

  const edge = new Float32Array(width * height);
  let maxEdge = 1;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const p = y * width + x;
      const gx = gray[p - 1] - gray[p + 1];
      const gy = gray[p - width] - gray[p + width];
      const e = Math.sqrt(gx * gx + gy * gy);
      edge[p] = e;
      if (e > maxEdge) maxEdge = e;
    }
  }

  // colored edge on black bg, then a cheap glow via box-blurred edge added back
  const glow = new Float32Array(width * height);
  const r = 2;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0, count = 0;
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            sum += edge[ny * width + nx];
            count++;
          }
        }
      }
      glow[y * width + x] = sum / count;
    }
  }

  for (let p = 0, i = 0; i < data.length; i += 4, p++) {
    const e = clamp((edge[p] / maxEdge) * 255 * 3);
    const gl = clamp((glow[p] / maxEdge) * 255 * 3);
    data[i] = clamp(e * 0.2 + gl * 0.5);
    data[i + 1] = clamp(e * 1.0 + gl * 0.6);
    data[i + 2] = clamp(e * 1.0 + gl * 0.9);
  }
}

function applyPinkGrid(imageData) {
  const { data, width, height } = imageData;
  const block = 14;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = clamp(data[i] * 0.5 + 255 * 0.5);
    data[i + 1] = clamp(data[i + 1] * 0.5 + 105 * 0.5);
    data[i + 2] = clamp(data[i + 2] * 0.5 + 180 * 0.5);
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (x % block === 0 || y % block === 0) {
        const idx = (y * width + x) * 4;
        data[idx] = 255; data[idx + 1] = 255; data[idx + 2] = 255;
      }
    }
  }
}

function clamp(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// --- MediaPipe wiring ---

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
});

hands.onResults(onResults);

const camera = new Camera(videoEl, {
  onFrame: async () => {
    await hands.send({ image: videoEl });
  },
  width: 1280,
  height: 720,
});

camera.start();
