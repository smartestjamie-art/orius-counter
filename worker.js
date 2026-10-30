/* Runs the counter off the page's main thread. Prepared photos stay cached by id, so
   changing thresholds only reruns detection. Every reply echoes the request's seq. */
importScripts('debris_model.js', 'orius.js');

const cache = new Map();

// Focus stacking: frames are blended with weight (sharpness + floor)^power per pixel, sharpness being |Laplacian| of
// the luma summed over a (2 r + 1)^2 window. In-focus detail comes almost wholly from the sharpest frame; flat
// background, where no frame is sharp, is averaged, so it neither flips between frames nor shows seams.
const STACK_BLEND = { power: 4, floor: 0.5 }; // floor: fraction of the first frame's median window sharpness
let stack = null;
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
      const rgba = new Uint8ClampedArray(m.rgba), score = stackScore(rgba, m.width, m.height, m.radius), n = m.width * m.height;
      if (!stack || stack.width !== m.width || stack.height !== m.height) {
        const sample = []; for (let i = 0; i < n; i += 97) sample.push(score[i]);
        sample.sort((a, b) => a - b);
        stack = { width: m.width, height: m.height, sum: new Float32Array(3 * n), weight: new Float32Array(n),
          floor: Math.max(1e-6, STACK_BLEND.floor * sample[sample.length >> 1]), frames: 0 };
      }
      const st = stack;
      for (let i = 0, j = 0; i < n; i++, j += 4) {
        const w = Math.pow(score[i] / st.floor + 1, STACK_BLEND.power);
        st.weight[i] += w; st.sum[3 * i] += w * rgba[j]; st.sum[3 * i + 1] += w * rgba[j + 1]; st.sum[3 * i + 2] += w * rgba[j + 2];
      }
      st.frames++;
      postMessage({ seq: m.seq, frames: st.frames });
    } else if (m.type === 'stackEnd') {
      const st = stack; stack = null;
      if (!st) { postMessage({ seq: m.seq, error: 'nothing stacked' }); return; }
      const n = st.width * st.height, out = new Uint8ClampedArray(4 * n);
      for (let i = 0, j = 0; i < n; i++, j += 4) {
        const w = st.weight[i];
        out[j] = st.sum[3 * i] / w; out[j + 1] = st.sum[3 * i + 1] / w; out[j + 2] = st.sum[3 * i + 2] / w; out[j + 3] = 255;
      }
      postMessage({ seq: m.seq, rgba: out.buffer, width: st.width, height: st.height, frames: st.frames }, [out.buffer]);
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
