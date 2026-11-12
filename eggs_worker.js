// Orius egg counter: finds the white egg caps on a green bean pod.
// Input: RGB at a working scale where an egg cap is about 4 px across (de = egg diameter there).
// 1. Pod mask: saturated green tissue (the white card is unsaturated), cleaned, holes filled, then pulled in
//    from the edge so the bright card next to the pod does not count.
// 2. Egg caps: pixels brighter AND bluer than their surroundings (green tissue has little blue, a white cap has
//    plenty), grouped into blobs; keep round blobs of egg size, drop long white slivers (hairs, fibres).
const VERSION = 'eggs-1';

self.onmessage = e => {
  const m = e.data;
  try {
    if (m.type === 'version') return self.postMessage({ seq: m.seq, version: VERSION });
    if (m.type === 'count') return self.postMessage({ seq: m.seq, ...count(m) });
    if (m.type === 'match') return self.postMessage({ seq: m.seq, ...match(m) });
  } catch (err) { self.postMessage({ seq: m.seq, error: err.message }); }
};

function count({ rgb, width: W, height: H, de, contrast, edge }) {
  const t0 = performance.now();
  const px = new Uint8Array(rgb), N = W * H;
  const L = new Float32Array(N), B = new Float32Array(N);
  let pod = new Uint8Array(N);
  for (let i = 0, j = 0; i < N; i++, j += 3) {
    const r = px[j], g = px[j + 1], b = px[j + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    L[i] = 0.299 * r + 0.587 * g + 0.114 * b; B[i] = b;
    pod[i] = (mx - mn) / (mx + 1) > 0.22 && g > b + 15 ? 1 : 0;
  }
  const r1 = Math.max(1, Math.round(de * 0.6)), r2 = Math.max(2, Math.round(de * 2));
  pod = dilate(erode(pod, W, H, r1), W, H, r1);       // opening: drop specks on the card
  pod = erode(dilate(pod, W, H, r2), W, H, r2);       // closing: bridge the seams
  fillHoles(pod, W, H);
  keepLargest(pod, W, H);
  const area = pod.reduce((a, v) => a + v, 0);
  pod = erode(pod, W, H, Math.max(1, Math.round(de * edge)));

  const win = Math.max(5, Math.round(de * 5)) | 1;
  const bL = boxMean(L, W, H, win >> 1), bB = boxMean(B, W, H, win >> 1);
  const cand = new Uint8Array(N), lift = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (!pod[i]) continue;
    const a = L[i] - bL[i], b = B[i] - bB[i];
    if (a > contrast && b > contrast) { cand[i] = 1; lift[i] = Math.min(a, b); }
  }

  // blobs, 8-connected
  const lab = new Int32Array(N), stack = [];
  const minA = Math.max(2, 0.25 * de * de), maxA = 4 * de * de, maxSide = 2.5 * de;
  const points = [], dropped = [];
  let next = 0;
  for (let s = 0; s < N; s++) {
    if (!cand[s] || lab[s]) continue;
    lab[s] = ++next; stack.push(s);
    let n = 0, sx = 0, sy = 0, x0 = W, x1 = 0, y0 = H, y1 = 0, peak = 0;
    while (stack.length) {
      const p = stack.pop(), x = p % W, y = (p - x) / W;
      n++; sx += x; sy += y; if (lift[p] > peak) peak = lift[p];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const X = x + dx, Y = y + dy;
        if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
        const q = Y * W + X;
        if (cand[q] && !lab[q]) { lab[q] = next; stack.push(q); }
      }
    }
    const cx = sx / n, cy = sy / n;
    if (n < minA) continue;                                           // noise
    if (n > maxA || Math.max(x1 - x0, y1 - y0) + 1 > maxSide) dropped.push([cx, cy]); // too big or long: hair, fibre, glare
    else points.push([cx, cy, Math.round(peak)]);
  }

  // pod outline, every few pixels, for drawing
  const outline = [];
  const step = Math.max(1, Math.round(Math.min(W, H) / 400));
  for (let y = 1; y < H - 1; y += step) for (let x = 1; x < W - 1; x += 1) {
    const i = y * W + x;
    if (pod[i] && (!pod[i - 1] || !pod[i + 1] || !pod[i - W] || !pod[i + W])) outline.push([x, y]);
  }
  for (let x = 1; x < W - 1; x += step) for (let y = 1; y < H - 1; y++) {
    const i = y * W + x;
    if (pod[i] && (!pod[i - W] || !pod[i + W])) outline.push([x, y]);
  }
  return { version: VERSION, points, dropped, outline, podArea: area, ms: Math.round(performance.now() - t0) };
}

// ---------- overlap between neighbouring shots ----------
// a, b: grey thumbnails (same size) of two shots along the rail. Finds the shift (s, dy) with b(x, y) ≈ a(x + s, y + dy):
// the tray only slides, so it is a plain shift. Texture (spots, seams, the pod outline) is matched after removing the
// lighting (high-pass) and clipping strong edges, so the pod outline does not outweigh everything else.
// Coarse search over every shift on a quarter-size copy, then refined at full thumbnail size.
// score: normalised correlation at the best shift; second: best score at a clearly different shift.
function match({ a, b, width: W, height: H }) {
  const A = texture(new Float32Array(a), W, H), B = texture(new Float32Array(b), W, H);
  const f = 4, w = Math.floor(W / f), h = Math.floor(H / f);
  const As = shrink(A, W, H, f), Bs = shrink(B, W, H, f);
  const minOv = Math.round(w * 0.06), maxDy = Math.round(h * 0.08);
  let best = { score: -2 }, scores = [];
  for (let s = -(w - minOv); s <= w - minOv; s++) {
    let top = -2, topDy = 0;
    for (let dy = -maxDy; dy <= maxDy; dy++) { const c = ncc(As, Bs, w, h, s, dy); if (c > top) { top = c; topDy = dy; } }
    scores.push([s, top]);
    if (top > best.score) best = { score: top, s, dy: topDy };
  }
  const second = Math.max(-1, ...scores.filter(([s]) => Math.abs(s - best.s) > 3).map(([, c]) => c));
  let fine = { score: -2 };
  for (let s = best.s * f - f - 1; s <= best.s * f + f + 1; s++)
    for (let dy = best.dy * f - f - 1; dy <= best.dy * f + f + 1; dy++) {
      if (Math.abs(s) > W - minOv * f) continue;
      const c = ncc(A, B, W, H, s, dy); if (c > fine.score) fine = { score: c, s, dy };
    }
  return { s: fine.s, dy: fine.dy, score: fine.score, second, coarse: best.score };
}

function texture(g, W, H) {
  const m = boxMean(g, W, H, 6), out = new Float32Array(W * H);
  let ss = 0;
  for (let i = 0; i < W * H; i++) { out[i] = g[i] - m[i]; ss += out[i] * out[i]; }
  const lim = 2.5 * Math.sqrt(ss / (W * H)) || 1;
  for (let i = 0; i < W * H; i++) out[i] = Math.max(-lim, Math.min(lim, out[i]));
  return out;
}
function shrink(v, W, H, f) {
  const w = Math.floor(W / f), h = Math.floor(H / f), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) s += v[(y * f + j) * W + x * f + i];
    out[y * w + x] = s / (f * f);
  }
  return out;
}
function ncc(A, B, W, H, s, dy) { // b pixel (x, y) against a pixel (x + s, y + dy), over the part both cover
  const x0 = Math.max(0, -s), x1 = Math.min(W, W - s), y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
  let n = 0, sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
  for (let y = y0; y < y1; y++) {
    const ra = (y + dy) * W + s, rb = y * W;
    for (let x = x0; x < x1; x++) { const p = A[ra + x], q = B[rb + x]; sa += p; sb += q; sab += p * q; saa += p * p; sbb += q * q; }
    n += x1 - x0;
  }
  if (n < 50) return -2;
  const cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : -2;
}

// Square min/max filters of radius r, separable, using a running count (binary images).
function erode(a, W, H, r) { return morph(a, W, H, r, true); }
function dilate(a, W, H, r) { return morph(a, W, H, r, false); }
function morph(a, W, H, r, isErode) {
  const tmp = new Uint8Array(W * H), out = new Uint8Array(W * H), need = 2 * r + 1;
  for (let y = 0; y < H; y++) {
    let c = 0; const o = y * W;
    for (let x = -r; x < W + r; x++) {
      const xi = x + r; if (xi < W && xi >= 0) c += a[o + xi]; else if (isErode) c += 0; // outside counts as empty
      const xo = x - r - 1; if (xo >= 0 && xo < W) c -= a[o + xo];
      if (x >= 0 && x < W) tmp[o + x] = isErode ? (c === need ? 1 : 0) : (c > 0 ? 1 : 0);
    }
  }
  for (let x = 0; x < W; x++) {
    let c = 0;
    for (let y = -r; y < H + r; y++) {
      const yi = y + r; if (yi < H && yi >= 0) c += tmp[yi * W + x];
      const yo = y - r - 1; if (yo >= 0 && yo < H) c -= tmp[yo * W + x];
      if (y >= 0 && y < H) out[y * W + x] = isErode ? (c === need ? 1 : 0) : (c > 0 ? 1 : 0);
    }
  }
  return out;
}

function fillHoles(a, W, H) { // background reachable from the border stays background, the rest becomes pod
  const seen = new Uint8Array(W * H), st = [];
  const push = i => { if (!a[i] && !seen[i]) { seen[i] = 1; st.push(i); } };
  for (let x = 0; x < W; x++) { push(x); push((H - 1) * W + x); }
  for (let y = 0; y < H; y++) { push(y * W); push(y * W + W - 1); }
  while (st.length) {
    const p = st.pop(), x = p % W, y = (p - x) / W;
    if (x > 0) push(p - 1); if (x < W - 1) push(p + 1); if (y > 0) push(p - W); if (y < H - 1) push(p + W);
  }
  for (let i = 0; i < W * H; i++) if (!seen[i]) a[i] = 1;
}

function keepLargest(a, W, H) { // keep pieces at least a fifth the size of the largest (a pod may be cut by the frame)
  const lab = new Int32Array(W * H), sizes = [0], st = [];
  for (let s = 0; s < W * H; s++) {
    if (!a[s] || lab[s]) continue;
    const id = sizes.length; let n = 0; lab[s] = id; st.push(s);
    while (st.length) {
      const p = st.pop(), x = p % W; n++;
      for (const q of [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, p - W, p + W])
        if (q >= 0 && q < W * H && a[q] && !lab[q]) { lab[q] = id; st.push(q); }
    }
    sizes.push(n);
  }
  const big = Math.max(...sizes);
  for (let i = 0; i < W * H; i++) if (a[i] && sizes[lab[i]] < big / 5) a[i] = 0;
}

function boxMean(v, W, H, r) { // mean over a (2r+1)^2 window, clipped at the edges
  const I = new Float64Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let row = 0;
    for (let x = 0; x < W; x++) { row += v[y * W + x]; I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + row; }
  }
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const ya = Math.max(0, y - r), yb = Math.min(H, y + r + 1);
    for (let x = 0; x < W; x++) {
      const xa = Math.max(0, x - r), xb = Math.min(W, x + r + 1);
      const s = I[yb * (W + 1) + xb] - I[ya * (W + 1) + xb] - I[yb * (W + 1) + xa] + I[ya * (W + 1) + xa];
      out[y * W + x] = s / ((yb - ya) * (xb - xa));
    }
  }
  return out;
}
