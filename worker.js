/* Runs the counter off the page's main thread. Prepared photos stay cached by id, so
   changing thresholds only reruns detection. Every reply echoes the request's seq. */
importScripts('debris_model.js', 'orius.js');

const cache = new Map();

// Focus stacking. Moving the focus changes the magnification a little (focus breathing), so each shot is first
// aligned to the reference shot: a scale about the frame centre plus a shift, found by matching the reference's
// strongest edges on 1/4-size gradient images (coarse grid, then fine). Then every area comes from the sharpest
// aligned shot: weight (sharpness + floor)^power, sharpness being |Laplacian| of the luma summed over a
// (2 r + 1)^2 window about the size of an insect, so an insect is taken whole from one shot and flat background,
// where no shot is sharp, is averaged.
const STACK_BLEND = { power: 8, floor: 0.5, window: 1 / 200, down: 4, samples: 8000,
  scale: [0.96, 1.04], coarseScale: 0.004, coarseShift: 2, maxShift: 4, fineScale: 0.0005, fineShift: 0.25 };
let stackFrames = [];

// Gradient magnitude of the luma, box-averaged down by `f`.
function smallGradient(rgba, W, H, f) {
  const w = Math.floor(W / f), h = Math.floor(H / f), g = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let dy = 0; dy < f; dy++) for (let dx = 0, j = 4 * ((y * f + dy) * W + x * f); dx < f; dx++, j += 4)
      acc += 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
    g[y * w + x] = acc / (f * f);
  }
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x; out[i] = Math.hypot(g[i + 1] - g[i - 1], g[i + w] - g[i - w]);
  }
  return { g: out, w, h };
}

function bilinear(img, w, h, x, y) {
  if (x < 0 || y < 0 || x > w - 2 || y > h - 2) return 0;
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * w + x0;
  return (img[i] * (1 - fx) + img[i + 1] * fx) * (1 - fy) + (img[i + w] * (1 - fx) + img[i + w + 1] * fx) * fy;
}

// Scale s (about the frame centre) and shift (tx, ty), in small-image pixels, mapping reference points into the frame.
function align(ref, frm) {
  const B = STACK_BLEND, { w, h } = ref, cx = w / 2, cy = h / 2;
  // the reference's strongest edges, away from the border
  const idx = [];
  for (let y = 12; y < h - 12; y++) for (let x = 12; x < w - 12; x++) idx.push(y * w + x);
  idx.sort((a, b) => ref.g[b] - ref.g[a]);
  const pts = idx.slice(0, B.samples).map(i => [i % w + 0.5 - cx, ((i / w) | 0) + 0.5 - cy, ref.g[i]]);
  const score = (s, tx, ty) => {
    let sum = 0;
    for (const [dx, dy, wt] of pts) sum += wt * bilinear(frm.g, w, h, cx + s * dx + tx - 0.5, cy + s * dy + ty - 0.5); // pixel centres at +0.5
    return sum;
  };
  let best = { s: 1, tx: 0, ty: 0, v: -1 };
  const tryAt = (s, tx, ty) => { const v = score(s, tx, ty); if (v > best.v) best = { s, tx, ty, v }; };
  for (let s = B.scale[0]; s <= B.scale[1] + 1e-9; s += B.coarseScale)
    for (let tx = -B.maxShift; tx <= B.maxShift; tx += B.coarseShift)
      for (let ty = -B.maxShift; ty <= B.maxShift; ty += B.coarseShift) tryAt(s, tx, ty);
  // refine scale, then shift, then scale again
  for (let round = 0; round < 2; round++) {
    const c = best;
    for (let s = c.s - B.coarseScale; s <= c.s + B.coarseScale + 1e-9; s += B.fineScale) tryAt(s, c.tx, c.ty);
    const d = best;
    for (let tx = d.tx - B.coarseShift; tx <= d.tx + B.coarseShift; tx += B.fineShift)
      for (let ty = d.ty - B.coarseShift; ty <= d.ty + B.coarseShift; ty += B.fineShift) tryAt(d.s, tx, ty);
  }
  return best;
}

// The frame resampled onto the reference's pixel grid (full size, Catmull-Rom bicubic, which keeps fine edges that
// bilinear interpolation softens); alpha 0 where it has no data.
function cubicWeights(t, w) {
  const t2 = t * t, t3 = t2 * t;
  w[0] = -0.5 * t3 + t2 - 0.5 * t; w[1] = 1.5 * t3 - 2.5 * t2 + 1; w[2] = -1.5 * t3 + 2 * t2 + 0.5 * t; w[3] = 0.5 * t3 - 0.5 * t2;
}
function warp(rgba, W, H, a, f) {
  const out = new Uint8ClampedArray(rgba.length), cx = W / 2, cy = H / 2, tx = a.tx * f, ty = a.ty * f;
  const wx = new Float64Array(4), wy = new Float64Array(4);
  for (let y = 0; y < H; y++) {
    const sy = cy + a.s * (y + 0.5 - cy) + ty - 0.5, y0 = Math.floor(sy);
    if (y0 < 1 || y0 > H - 3) continue;
    cubicWeights(sy - y0, wy);
    for (let x = 0; x < W; x++) {
      const sx = cx + a.s * (x + 0.5 - cx) + tx - 0.5, x0 = Math.floor(sx);
      if (x0 < 1 || x0 > W - 3) continue;
      cubicWeights(sx - x0, wx);
      const o = 4 * (y * W + x);
      for (let c = 0; c < 3; c++) {
        let acc = 0;
        for (let j = 0; j < 4; j++) {
          const row = 4 * ((y0 - 1 + j) * W + x0 - 1) + c;
          acc += wy[j] * (wx[0] * rgba[row] + wx[1] * rgba[row + 4] + wx[2] * rgba[row + 8] + wx[3] * rgba[row + 12]);
        }
        out[o + c] = acc;
      }
      out[o + 3] = 255;
    }
  }
  return out;
}
function stackScore(rgba, w, h, r) {
  const g = new Float32Array(w * h), lap = new Float32Array(w * h);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  for (let y = 1; y < h - 1; y++) for (let x = 1, i = y * w + 1; x < w - 1; x++, i++)
    lap[i] = Math.abs(4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w]);
  // box sum: rows, then columns, with running sums
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    let acc = 0; const o = y * w;
    for (let x = -r; x < w; x++) {
      if (x + r < w) acc += lap[o + x + r];
      if (x - r - 1 >= 0) acc -= lap[o + x - r - 1];
      if (x >= 0) tmp[o + x] = acc;
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y < h; y++) {
      if (y + r < h) acc += tmp[(y + r) * w + x];
      if (y - r - 1 >= 0) acc -= tmp[(y - r - 1) * w + x];
      if (y >= 0) out[y * w + x] = acc;
    }
  }
  return out;
}

onmessage = e => {
  const m = e.data;
  try {
    if (m.type === 'prepare') {
      const prep = Orius.prepare(new Uint8Array(m.rgb), m.width, m.height);
      cache.set(m.id, prep);
      postMessage({ seq: m.seq, dish: prep.dish, width: prep.width, height: prep.height });
    } else if (m.type === 'count') {
      const prep = cache.get(m.id);
      const t = performance.now();
      const r = Orius.count(prep, { gray: m.gray, core: m.core, nymphs: m.nymphs });
      // cands: [x, y, p_insect, dropped (0 kept, 1 straw rule, 2 classifier), 1 for a nymph]; nymphs come last; feats: the classifier's inputs, for corrections
      const names = self.ORIUS_DEBRIS_MODEL ? ORIUS_DEBRIS_MODEL.features : [];
      postMessage({ seq: m.seq, count: r.count, points: r.detections.map(d => [d.x, d.y]),
        cands: r.candidates.map(d => [d.x, d.y, d.p_insect == null ? 1 : d.p_insect, d.dropped === 'straw' ? 1 : d.dropped ? 2 : 0, 0])
          .concat(r.detections.filter(d => d.kind === 'nymph').map(d => [d.x, d.y, 1, 0, 1])),
        feats: r.candidates.map(d => d.features ? names.map(k => d.features[k]) : null).concat(Array(r.nymphs).fill(null)), featureNames: names,
        debrisRemoved: r.debrisRemoved + r.classifierRemoved, backlit: r.backlit, version: Orius.VERSION, nymphs: r.nymphs, ms: Math.round(performance.now() - t) });
    } else if (m.type === 'live') {
      // A preview frame: prepared and counted once, not cached; points as fractions of the frame.
      const prep = Orius.prepare(new Uint8Array(m.rgb), m.width, m.height);
      const r = Orius.count(prep, { gray: m.gray, core: m.core, nymphs: m.nymphs });
      postMessage({ seq: m.seq, count: r.count, points: r.detections.map(d => [d.x / prep.width, d.y / prep.height, d.kind === 'nymph' ? 1 : 0]) });
    } else if (m.type === 'stackAdd') {
      stackFrames.push({ rgba: new Uint8ClampedArray(m.rgba), width: m.width, height: m.height });
      postMessage({ seq: m.seq, frames: stackFrames.length });
    } else if (m.type === 'stackEnd') {
      const frames = stackFrames; stackFrames = [];
      if (!frames.length) { postMessage({ seq: m.seq, error: 'nothing stacked' }); return; }
      const B = STACK_BLEND, W = frames[0].width, H = frames[0].height, n = W * H, f = B.down;
      const ref = Math.min(frames.length - 1, Math.max(0, m.ref == null ? frames.length >> 1 : m.ref));
      const refG = smallGradient(frames[ref].rgba, W, H, f), radius = Math.max(2, Math.round(W * B.window));
      const sum = new Float32Array(3 * n), weight = new Float32Array(n), shifts = [];
      let floor = 0;
      for (const k of [ref, ...frames.keys()].filter((k, i, all) => all.indexOf(k) === i)) {
        const a = k === ref ? { s: 1, tx: 0, ty: 0 } : align(refG, smallGradient(frames[k].rgba, W, H, f));
        shifts[k] = { scale: +a.s.toFixed(4), dx: +(a.tx * f).toFixed(1), dy: +(a.ty * f).toFixed(1) };
        const img = k === ref ? frames[k].rgba : warp(frames[k].rgba, W, H, a, f);
        frames[k].rgba = null;
        const score = stackScore(img, W, H, radius);
        if (!floor) {
          const sample = []; for (let i = 0; i < n; i += 97) sample.push(score[i]);
          sample.sort((x, y) => x - y); floor = Math.max(1e-6, B.floor * sample[sample.length >> 1]);
        }
        for (let i = 0, j = 0; i < n; i++, j += 4) {
          if (!img[j + 3]) continue; // no data here after alignment
          const wt = Math.pow(score[i] / floor + 1, B.power);
          weight[i] += wt; sum[3 * i] += wt * img[j]; sum[3 * i + 1] += wt * img[j + 1]; sum[3 * i + 2] += wt * img[j + 2];
        }
      }
      const out = new Uint8ClampedArray(4 * n);
      for (let i = 0, j = 0; i < n; i++, j += 4) {
        const w = weight[i] || 1;
        out[j] = sum[3 * i] / w; out[j + 1] = sum[3 * i + 1] / w; out[j + 2] = sum[3 * i + 2] / w; out[j + 3] = 255;
      }
      postMessage({ seq: m.seq, rgba: out.buffer, width: W, height: H, frames: frames.length, shifts }, [out.buffer]);
    } else if (m.type === 'version') {
      postMessage({ seq: m.seq, version: Orius.VERSION });
    } else if (m.type === 'drop') {
      cache.delete(m.id);
      postMessage({ seq: m.seq });
    }
  } catch (err) {
    postMessage({ seq: m.seq, error: String((err && err.message) || err) });
  }
};
