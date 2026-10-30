/* Runs the counter off the page's main thread. Prepared photos stay cached by id, so
   changing thresholds only reruns detection. Every reply echoes the request's seq. */
importScripts('debris_model.js', 'orius.js');

const cache = new Map();

// Focus stacking: each added frame replaces the pixels where it is sharper than every frame so far. Sharpness is
// |Laplacian| of the luma summed over a (2 r + 1)^2 window, so whole insects come from one frame rather than speckle.
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
      const rgba = new Uint8ClampedArray(m.rgba), score = stackScore(rgba, m.width, m.height, m.radius);
      if (!stack || stack.width !== m.width || stack.height !== m.height) {
        stack = { width: m.width, height: m.height, rgba, best: score, frames: 1 };
      } else {
        for (let i = 0; i < score.length; i++) if (score[i] > stack.best[i]) {
          stack.best[i] = score[i]; const j = 4 * i;
          stack.rgba[j] = rgba[j]; stack.rgba[j + 1] = rgba[j + 1]; stack.rgba[j + 2] = rgba[j + 2];
        }
        stack.frames++;
      }
      postMessage({ seq: m.seq, frames: stack.frames });
    } else if (m.type === 'stackEnd') {
      const st = stack; stack = null;
      if (!st) { postMessage({ seq: m.seq, error: 'nothing stacked' }); return; }
      postMessage({ seq: m.seq, rgba: st.rgba.buffer, width: st.width, height: st.height, frames: st.frames }, [st.rgba.buffer]);
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
