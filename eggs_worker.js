// Orius egg counter: finds the white egg caps on a green bean pod.
// Input: RGB at a working scale where an egg cap is about 4 px across (de = egg diameter there).
// 1. Pod mask: saturated green tissue (the white card is unsaturated), cleaned, holes filled, then pulled in
//    from the edge so the bright card next to the pod does not count.
// 2. Egg caps: pixels brighter AND bluer than their surroundings (green tissue has little blue, a white cap has
//    plenty), grouped into blobs; keep round blobs of egg size, drop long white slivers (hairs, fibres).
const VERSION = 'eggs-3'; // counting; overlap matching changed separately (no recount needed)

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
  const neutral = new Uint8Array(N);
  for (let i = 0, j = 0; i < N; i++, j += 3) {
    const r = px[j], g = px[j + 1], b = px[j + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    L[i] = 0.299 * r + 0.587 * g + 0.114 * b; B[i] = b;
    // pod: saturated green, or pale green (the pod turns whitish where the fuzz catches the light, often where eggs are
    // densest: on the rail rig, pale pod is about RGB 133,165,152 while the card is neutral, about 195,194,198)
    pod[i] = ((mx - mn) / (mx + 1) > 0.22 && g > b + 15) || (g > r + 12 && g > b + 3 && L[i] > 50) ? 1 : 0;
    neutral[i] = L[i] > 140 && Math.abs(g - r) < 10 && Math.abs(g - b) < 12 ? 1 : 0; // white card
  }
  const r1 = Math.max(1, Math.round(de * 0.6)), r2 = Math.max(2, Math.round(de * 2));
  pod = dilate(erode(pod, W, H, r1), W, H, r1);       // opening: drop specks on the card
  pod = erode(dilate(pod, W, H, r2), W, H, r2);       // closing: bridge the seams
  fillHoles(pod, W, H);
  keepLargest(pod, W, H);
  const area = pod.reduce((a, v) => a + v, 0);
  // Counted zone: the pod and a margin past its edge (edge, in egg sizes), so eggs on the brown seam along the rim
  // count too; the card itself stays out (large bright neutral areas: an opening keeps them and drops egg-sized
  // specks), and so does the dark background.
  const rc = Math.max(2, Math.round(de * 0.9));
  const card = dilate(erode(neutral, W, H, rc), W, H, rc);
  const zone = dilate(pod, W, H, Math.max(0, Math.round(de * edge)));
  // and a margin at the picture's border (a stacked photo has a ragged edge where the frames were shifted)
  const mg = Math.max(4, Math.round(de * 2));
  for (let i = 0; i < N; i++) { const x = i % W, y = (i - x) / W; if (card[i] || L[i] < 45 || x < mg || y < mg || x >= W - mg || y >= H - mg) zone[i] = 0; }
  pod = zone;

  // local background from the zone only, so the bright card next to the rim does not raise it
  const win = Math.max(5, Math.round(de * 5)) | 1;
  const zf = new Float32Array(N), zL = new Float32Array(N), zB = new Float32Array(N);
  for (let i = 0; i < N; i++) if (zone[i]) { zf[i] = 1; zL[i] = L[i]; zB[i] = B[i]; }
  const mZ = boxMean(zf, W, H, win >> 1), mL = boxMean(zL, W, H, win >> 1), mB = boxMean(zB, W, H, win >> 1);
  const cand = new Uint8Array(N), lift = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (!zone[i] || mZ[i] < 0.2) continue;
    const a = L[i] - mL[i] / mZ[i], b = B[i] - mB[i] / mZ[i];
    if (a > contrast && b > contrast) { cand[i] = 1; lift[i] = Math.min(a, b); }
  }

  // blobs, 8-connected
  const lab = new Int32Array(N), stack = [];
  const minA = Math.max(2, 0.25 * de * de), maxA = 4 * de * de, maxSide = 2.5 * de;
  const points = [], dropped = [], big = [];
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
    if (n > maxA || Math.max(x1 - x0, y1 - y0) + 1 > maxSide) big.push({ cx, cy, n, w: x1 - x0 + 1, h: y1 - y0 + 1, peak }); // decided below
    else points.push([cx, cy, Math.round(peak), n]);
  }
  // Too big or too long for one egg: hairs, fibres, glare, debris and shed skins. (Counting compact ones as clumps of
  // eggs by area was tried on Pod 6 A3: most of them were pale debris, not eggs, so they are dropped; touching eggs
  // can be added by hand.)
  for (const b of big) dropped.push([b.cx, b.cy]);
  for (const p of points) p.length = 3;

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
// Only the pod is matched (ma, mb: green-pod masks): the camera, rail, background and lighting stay put while the
// tray slides, so they would pull the match towards no shift at all.
// With pod masks: candidates from the pod's outline (its wavy edges and ends: the pod moves 9-14 % up or down between
// marks on the rail rig, and plain texture also matches things that stay put), each confirmed by the pod's surface
// texture; score = texture x shape. Without masks, the texture search below.
function match(m) {
  if (m.ma && m.mb && !self.MATCH_NOSTRIP) { const r = matchStrip(m); if (r) return r; }
  if (m.ma && m.mb && !self.MATCH_OLD) { const r = matchShape(m); if (r) return r; }
  return matchTexture(m);
}
// Tray-edge strips (print/tray_edge_strips.pdf: random black dots, 1.2-3.2 mm, on white, stuck on the tray so they
// move with the pod). Each shot's dots are found (small, compact, dark on a bright surround, off the pod), and every
// pair of similar-sized dots, one per shot, votes for the shift between them: the true shift gets a vote from every
// dot the two shots share, any other only from chance pairs. Things that stay put (rail, screws, vignetting) can only
// vote near no shift, which the tray never makes. Used only when the vote is clear; otherwise the pod is matched.
function stripDots(g, raw, W, H) {
  const mean = boxMean(g, W, H, 16), pod = dilate(podPiece(raw, g, W, H), W, H, 6), d = new Uint8Array(W * H);
  for (let i = 0; i < d.length; i++) d[i] = !pod[i] && mean[i] > 90 && g[i] < 0.62 * mean[i] ? 1 : 0;
  const lab = new Int32Array(W * H), stack = new Int32Array(W * H), dots = [];
  for (let i = 0; i < d.length; i++) {
    if (!d[i] || lab[i]) continue;
    let n = 0, sx = 0, sy = 0, x0 = W, x1 = 0, y0 = H, y1 = 0, sp = 0; stack[sp++] = i; lab[i] = 1;
    while (sp) {
      const p = stack[--sp], x = p % W, y = (p - x) / W; n++; sx += x; sy += y;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (const q of [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, y > 0 ? p - W : -1, y < H - 1 ? p + W : -1])
        if (q >= 0 && d[q] && !lab[q]) { lab[q] = 1; stack[sp++] = q; }
    }
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1, edge = x0 === 0 || y0 === 0 || x1 === W - 1 || y1 === H - 1;
    // a dot: 3 px up to about 4 mm across, round-ish (fills most of its box), not cut by the frame; lines, ruler
    // ticks, numbers and dark edges are left out
    if (!edge && n >= 3 && bw <= 40 && bh <= 40 && bw <= 2 * bh + 1 && bh <= 2 * bw + 1 && n >= 0.5 * bw * bh) dots.push([sx / n, sy / n, n]);
  }
  return dots;
}
function matchStrip({ a, b, ma, mb, width: W, height: H }) {
  const ga = new Float32Array(a), gb = new Float32Array(b);
  const A = stripDots(ga, new Uint8Array(ma), W, H), B = stripDots(gb, new Uint8Array(mb), W, H);
  if (A.length < 15 || B.length < 15) return null;
  const sAway = W * 0.08, sMax = W * 0.88, dyMax = H * 0.2, bin = 3, nx = Math.ceil(W / bin), ny = Math.ceil(2 * dyMax / bin) + 1;
  const votes = new Float32Array(nx * ny);
  for (const [xa, ya, na] of A) for (const [xb, yb, nb] of B) {
    if (na > 1.6 * nb || nb > 1.6 * na) continue;
    const s = xa - xb, dy = ya - yb;
    if (s < sAway || s > sMax || Math.abs(dy) > dyMax) continue;
    votes[Math.floor((dy + dyMax) / bin) * nx + Math.floor(s / bin)]++;
  }
  // votes summed over 3 x 3 bins (a shift on a bin's border splits its votes)
  const sum = (bx, by) => { let t = 0; for (let j = by - 1; j <= by + 1; j++) for (let i = bx - 1; i <= bx + 1; i++) if (i >= 0 && j >= 0 && i < nx && j < ny) t += votes[j * nx + i]; return t; };
  // the 15 best-voted shifts (chance pairs among ~150 dots give every shift a few votes, so the votes only shortlist)
  const cands = [];
  for (let by = 0; by < ny; by++) for (let bx = 0; bx < nx; bx++) if (votes[by * nx + bx] >= 2) cands.push({ v: sum(bx, by), bx, by });
  cands.sort((u, v) => v.v - u.v);
  const top = [];
  for (const c of cands) { if (top.length >= 15) break; if (top.every(t => Math.abs(t.bx - c.bx) > 2 || Math.abs(t.by - c.by) > 2)) top.push(c); }
  if (!top.length) return null;
  // each checked dot by dot: the share of A's dots inside the overlap that land on a B dot of similar size, one to one
  const cell = 8, gw = Math.ceil(W / cell), grid = new Map();
  B.forEach((d, j) => { const k = Math.floor(d[1] / cell) * gw + Math.floor(d[0] / cell); (grid.get(k) || grid.set(k, []).get(k)).push(j); });
  const med = v => { const t = v.sort((x, y) => x - y); return t[t.length >> 1]; };
  const check = (s, dy, rad) => {
    const used = new Set(), ss = [], ds = []; let inside = 0;
    for (const [xa, ya, na] of A) {
      const x = xa - s, y = ya - dy;
      if (x < 4 || y < 4 || x > W - 5 || y > H - 5) continue;
      inside++;
      let bj = -1, bd = rad * rad;
      for (let gy = Math.floor(y / cell) - 1; gy <= Math.floor(y / cell) + 1; gy++) for (let gx = Math.floor(x / cell) - 1; gx <= Math.floor(x / cell) + 1; gx++)
        for (const j of grid.get(gy * gw + gx) || []) {
          const [xb, yb, nb] = B[j], e = (xb - x) ** 2 + (yb - y) ** 2;
          if (e < bd && !used.has(j) && na <= 1.5 * nb && nb <= 1.5 * na) { bd = e; bj = j; }
        }
      if (bj >= 0) { used.add(bj); ss.push(xa - B[bj][0]); ds.push(ya - B[bj][1]); }
    }
    return { n: ss.length, inside, s: ss.length ? med(ss) : s, dy: ds.length ? med(ds) : dy };
  };
  const res = top.map(c => { // loosely around the voted bin first, then tightly around the shift those dots give
    let r = check((c.bx + 0.5) * bin, (c.by + 0.5) * bin - dyMax, 3); r = check(r.s, r.dy, 1.6); return check(r.s, r.dy, 1.6); }).sort((u, v) => v.n - u.n);
  const best = res[0], second = Math.max(0, ...res.filter(r => Math.abs(r.s - best.s) > 6 || Math.abs(r.dy - best.dy) > 6).map(r => r.n));
  // clear: ten shared dots at least, a third of the dots in the overlap matched, and twice any other shift
  if (best.n < 10 || best.n < 0.33 * best.inside || best.n < 2 * second) return null;
  return { s: Math.round(best.s), dy: Math.round(best.dy), score: best.n / 100, second: second / 100, dots: [A.length, B.length], shared: best.n, inside: best.inside, strip: true, combined: true };
}
function matchShape({ a, b, ma, mb, width: W, height: H, near }) {
  const A = texture(new Float32Array(a), W, H), B = texture(new Float32Array(b), W, H);
  // The green mask is speckled wherever eggs and brown patches are dense, and podMask's erosion then wipes out most of
  // the real pod, leaving solid false patches (vignetted card corners and edges, the dark tray end) as all the "pod"
  // two shots share: on the 'test' pod's A3|A4 the texture was compared on A3's vignetted right edge and a join 29 %
  // of a frame off won. So the speckles are closed first, and only the main pod piece, not dark, is kept.
  const rawA = podPiece(new Uint8Array(ma), new Float32Array(a), W, H), rawB = podPiece(new Uint8Array(mb), new Float32Array(b), W, H);
  const MA = podMask(rawA, W, H), MB = podMask(rawB, W, H);
  const shapeOf = raw => { const o = dilate(erode(raw, W, H, 2), W, H, 2), f = new Float32Array(W * H); for (let i = 0; i < f.length; i++) f[i] = o[i]; return boxMean(f, W, H, 4); };
  const q = 4, wq = Math.floor(W / q), hq = Math.floor(H / q), SA = shrink(shapeOf(rawA), W, H, q), SB = shrink(shapeOf(rawB), W, H, q);
  const podN = v => v.reduce((t, x) => t + x, 0);
  if (podN(MA) < 0.01 * W * H || podN(MB) < 0.01 * W * H) return null;
  // overlaps on the rigs are 14-76 % of the width; slivers under 12 % match noise, so they are not considered
  const minOv = Math.round(wq * 0.12), sAway = Math.round(wq * 0.08), maxDy = Math.round(hq * (self.MATCH_DY || 0.2));
  // the tray moves the same way between every pair of marks (each shot shows the pod further along): positive shifts only
  const sLo = near != null ? Math.round((near - 0.06) * wq) : sAway, sHi = near != null ? Math.round((near + 0.06) * wq) : wq - minOv;
  // 1. every shift (2 px steps at half size; the tray always moves, so none near zero: that is dust and shading): the
  // outline at quarter size first, and where it agrees at all, the surface texture at half size
  const h2 = 2, w2 = Math.floor(W / h2), hh2 = Math.floor(H / h2), A2 = shrink(A, W, H, h2), B2 = shrink(B, W, H, h2), MA2 = shrinkMask(MA, W, H, h2), MB2 = shrinkMask(MB, W, H, h2);
  // shared pod needed: a sixteenth of the smaller shot's pod (overlaps on the rig can be as narrow as 14 %)
  const need = Math.max(30, 0.0625 * Math.min(podN(MA2), podN(MB2)));
  const scored = [];
  for (let s = Math.max(sLo, -(wq - minOv)); s <= Math.min(sHi, wq - minOv); s++) {
    if (Math.abs(s) < sAway) continue;
    for (let dy = -maxDy; dy <= maxDy; dy++) {
      const c = nccPlain(SA, SB, wq, hq, s, dy); if (c < 0.3) continue;
      const t = nccMasked(A2, B2, MA2, MB2, w2, hh2, 2 * s, 2 * dy, need); if (t <= 0) continue;
      scored.push({ c, tex: t, s, dy, s2: 2 * s, dy2: 2 * dy, score: t * c });
    }
  }
  if (!scored.length) return null;
  scored.sort((u, v) => v.score - u.score);
  // refine the best at half size, 1 px steps
  let best = scored[0];
  for (let s = best.s2 - 2; s <= best.s2 + 2; s++) for (let dy = best.dy2 - 2; dy <= best.dy2 + 2; dy++) {
    const t = nccMasked(A2, B2, MA2, MB2, w2, hh2, s, dy, need), c = nccPlain(SA, SB, wq, hq, Math.round(s / 2), Math.round(dy / 2));
    if (t * c > best.score) best = { ...best, tex: t, c, s2: s, dy2: dy, score: t * c };
  }
  const second = Math.max(0, ...scored.filter(c => Math.abs(c.s - best.s) > Math.round(wq * 0.05)).map(c => c.score));
  // 3. fine position on the full thumbnail
  const needF = Math.max(50, 0.0625 * Math.min(podN(MA), podN(MB)));
  let fine = { v: -2, s: best.s2 * 2, dy: best.dy2 * 2 };
  for (let s = best.s2 * 2 - 2; s <= best.s2 * 2 + 2; s++) for (let dy = best.dy2 * 2 - 2; dy <= best.dy2 * 2 + 2; dy++) {
    const v = nccMasked(A, B, MA, MB, W, H, s, dy, needF); if (v > fine.v) fine = { v, s, dy };
  }
  return { s: fine.s, dy: fine.dy, score: best.score, second, coarse: best.score, tex: best.tex, shape: best.c, masked: true, combined: true };
}
function matchTexture({ a, b, ma, mb, width: W, height: H, near }) {
  const A = texture(new Float32Array(a), W, H), B = texture(new Float32Array(b), W, H);
  // half-size search: at quarter size the pod's fine surface texture is gone and the match lands on the wrong shift
  // (and coarser for big or tall thumbnails, so the search never takes more than a few seconds)
  const f = +(self.MATCH_F || Math.max(2, Math.ceil(Math.sqrt(W * H / 80000)))), w = Math.floor(W / f), h = Math.floor(H / f);
  const As = shrink(A, W, H, f), Bs = shrink(B, W, H, f);
  let MA = null, MB = null, MAs = null, MBs = null;
  if (ma && mb) {
    MA = podMask(new Uint8Array(ma), W, H); MB = podMask(new Uint8Array(mb), W, H);
    MAs = shrinkMask(MA, W, H, f); MBs = shrinkMask(MB, W, H, f);
    // too little pod in either shot: match the whole picture as before
    const share = v => v.reduce((t, x) => t + x, 0) / v.length;
    if (share(MAs) < 0.01 || share(MBs) < 0.01) MA = MB = MAs = MBs = null;
  }
  // shared pod needed: an eighth of the smaller shot's pod (a thin pod and a narrow overlap still qualify)
  const podN = v => v ? v.reduce((t, x) => t + x, 0) : 0;
  const needFull = MA ? Math.max(50, 0.125 * Math.min(podN(MA), podN(MB))) : 0, needSmall = MAs ? Math.max(30, 0.125 * Math.min(podN(MAs), podN(MBs))) : 0;
  const ncc = (P, Q, ww, hh, s, dy) => MAs ? nccMasked(P, Q, ww === W ? MA : MAs, ww === W ? MB : MBs, ww, hh, s, dy, ww === W ? needFull : needSmall) : nccPlain(P, Q, ww, hh, s, dy);
  const minOv = Math.round(w * 0.06), maxDy = Math.round(h * (self.MATCH_DY || 0.08)); // the tray slides along the rail, but A1-A2 on the real rig moved 6 % down
  // near: the shift this pair of rail marks is known to have (fraction of the width): search only close to it
  const sLo = near != null ? Math.round((near - 0.06) * w) : -(w - minOv), sHi = near != null ? Math.round((near + 0.06) * w) : w - minOv;
  let best = { score: -2 }, scores = [];
  for (let s = Math.max(sLo, -(w - minOv)); s <= Math.min(sHi, w - minOv); s++) {
    let top = -2, topDy = 0;
    for (let dy = -maxDy; dy <= maxDy; dy++) { const c = ncc(As, Bs, w, h, s, dy); if (c > top) { top = c; topDy = dy; } }
    scores.push([s, top]);
    if (top > best.score) best = { score: top, s, dy: topDy };
  }
  // the best score away from the peak (its shoulders are broad: pod texture changes slowly), for the clarity test
  const away = Math.max(3, Math.round(w * 0.05));
  const second = Math.max(-1, ...scores.filter(([s]) => Math.abs(s - best.s) > away).map(([, c]) => c));
  let fine = { score: -2 };
  for (let s = best.s * f - f - 1; s <= best.s * f + f + 1; s++)
    for (let dy = best.dy * f - f - 1; dy <= best.dy * f + f + 1; dy++) {
      if (Math.abs(s) > W - minOv * f) continue;
      const c = ncc(A, B, W, H, s, dy); if (c > fine.score) fine = { score: c, s, dy };
    }
  return { s: fine.s, dy: fine.dy, score: fine.score, second, coarse: best.score, masked: !!MAs };
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
// The pod as one piece: green and not dark (the tray end and deep vignetting are dark), speckles closed, the largest
// piece only.
function podPiece(raw, lum, W, H) {
  const m = new Uint8Array(W * H);
  for (let i = 0; i < m.length; i++) m[i] = raw[i] && lum[i] > 45 ? 1 : 0;
  const c = erode(dilate(m, W, H, 3), W, H, 3);
  const lab = new Int32Array(W * H), stack = new Int32Array(W * H);
  let best = 0, bestN = 0, id = 0;
  for (let i = 0; i < c.length; i++) {
    if (!c[i] || lab[i]) continue;
    id++; let n = 0, sp = 0; stack[sp++] = i; lab[i] = id;
    while (sp) {
      const p = stack[--sp], x = p % W; n++;
      if (x > 0 && c[p - 1] && !lab[p - 1]) { lab[p - 1] = id; stack[sp++] = p - 1; }
      if (x < W - 1 && c[p + 1] && !lab[p + 1]) { lab[p + 1] = id; stack[sp++] = p + 1; }
      if (p >= W && c[p - W] && !lab[p - W]) { lab[p - W] = id; stack[sp++] = p - W; }
      if (p < W * (H - 1) && c[p + W] && !lab[p + W]) { lab[p + W] = id; stack[sp++] = p + W; }
    }
    if (n > bestN) { bestN = n; best = id; }
  }
  const out = new Uint8Array(W * H);
  for (let i = 0; i < out.length; i++) out[i] = best && lab[i] === best ? 1 : 0;
  return out;
}
// pod in both shots: the mask cleaned of specks and slightly shrunk, so the pod's own texture is matched, not its rim
function podMask(m, W, H) { return erode(dilate(erode(m, W, H, 1), W, H, 1), W, H, 2); }
function shrinkMask(m, W, H, f) {
  const w = Math.floor(W / f), h = Math.floor(H / f), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) s += m[(y * f + j) * W + x * f + i];
    out[y * w + x] = s * 2 > f * f ? 1 : 0;
  }
  return out;
}
// as nccPlain, over pixels that are pod in both shots; needs a fair amount of shared pod
function nccMasked(A, B, MA, MB, W, H, s, dy, need) {
  const x0 = Math.max(0, -s), x1 = Math.min(W, W - s), y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
  let n = 0, sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
  for (let y = y0; y < y1; y++) {
    const ra = (y + dy) * W + s, rb = y * W;
    for (let x = x0; x < x1; x++) {
      if (!MA[ra + x] || !MB[rb + x]) continue;
      const p = A[ra + x], q = B[rb + x]; sa += p; sb += q; sab += p * q; saa += p * p; sbb += q * q; n++;
    }
  }
  if (n < need) return -2;
  const cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : -2;
}
function nccPlain(A, B, W, H, s, dy) { // b pixel (x, y) against a pixel (x + s, y + dy), over the part both cover
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
