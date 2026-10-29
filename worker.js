/* Runs the counter off the page's main thread. Prepared photos stay cached by id, so
   changing thresholds only reruns detection. Every reply echoes the request's seq. */
importScripts('debris_model.js', 'orius.js');

const cache = new Map();

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
      const r = Orius.count(prep, { gray: m.gray, core: m.core });
      // cands: [x, y, p_insect, dropped (0 kept, 1 straw rule, 2 classifier)]; feats: the classifier's inputs, for corrections
      const names = self.ORIUS_DEBRIS_MODEL ? ORIUS_DEBRIS_MODEL.features : [];
      postMessage({ seq: m.seq, count: r.count, points: r.detections.map(d => [d.x, d.y]),
        cands: r.candidates.map(d => [d.x, d.y, d.p_insect == null ? 1 : d.p_insect, d.dropped === 'straw' ? 1 : d.dropped ? 2 : 0]),
        feats: r.candidates.map(d => d.features ? names.map(k => d.features[k]) : null), featureNames: names,
        debrisRemoved: r.debrisRemoved + r.classifierRemoved, backlit: r.backlit, ms: Math.round(performance.now() - t) });
    } else if (m.type === 'live') {
      // A preview frame: prepared and counted once, not cached; points as fractions of the frame.
      const prep = Orius.prepare(new Uint8Array(m.rgb), m.width, m.height);
      const r = Orius.count(prep, { gray: m.gray, core: m.core });
      postMessage({ seq: m.seq, count: r.count, points: r.detections.map(d => [d.x / prep.width, d.y / prep.height]) });
    } else if (m.type === 'drop') {
      cache.delete(m.id);
      postMessage({ seq: m.seq });
    }
  } catch (err) {
    postMessage({ seq: m.seq, error: String((err && err.message) || err) });
  }
};
