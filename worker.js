/* Runs the counter off the page's main thread. Prepared photos stay cached by id, so
   changing thresholds only reruns detection. Every reply echoes the request's seq. */
importScripts('debris_model.js', 'orius.js');

const cache = new Map();

// Focus stacking, after the stack.py / focus-stack approach.
// Frames arrive as JPEG blobs with their focus value. Each is aligned to the middle frame (scale about the centre +
// shift: coarse on 1/4-size gradients, sub-pixel on 1/2 size; focus breathing changes the scale smoothly with the
// focus value, so scales off a straight-line fit are replaced by the fit), and its tones are matched to the
// reference (auto exposure can jump between frames, which shows as blotches wherever the merge switches frames).
// Sharpness is Laplacian^2 on half-size copies (noise averages out), Gaussian-smoothed. Where some frame has real
// detail the sharpest frame is chosen; the choice map is median-filtered and spread into a smooth depth surface, and
// each pixel blends only the frames next to its depth. Flat background comes from the single sharpest frame.
const STACK = { coarseScale: 0.004, scaleRange: 0.04, coarseShift: 2, maxShift: 4, samples: 8000,
  fineScale: 0.0005, fineShift: 0.25, halfScale: 0.0001, halfShift: 0.25, lineTol: 0.001,
  energySigma: 4, detail: 1.5, strong: 4, median: 3, wideSigma: 10, keepDilate: 7, keepSigma: 6, power: 6, reach: 0.6 };
let stackFrames = [];

async function decodeBlob(blob, w, h) {
  const bmp = await createImageBitmap(blob);
  const W = w || bmp.width, H = h || bmp.height, oc = new OffscreenCanvas(W, H), ctx = oc.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, 0, 0, W, H); bmp.close();
  return { rgba: ctx.getImageData(0, 0, W, H).data, w: W, h: H };
}

function lumaOf(rgba, n) {
  const g = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) g[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  return g;
}
function gradientOf(rgba, w, h) {
  const g = lumaOf(rgba, w * h), out = new Float32Array(w * h);
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

// Scale about the image centre and shift (in this image's pixels) mapping reference points into the frame.
function edgePoints(ref, count) {
  const { w, h, g } = ref, bins = 1024;
  let max = 0;
  for (let i = 0; i < g.length; i++) if (g[i] > max) max = g[i];
  const hist = new Int32Array(bins);
  for (let y = 12; y < h - 12; y++) for (let x = 12; x < w - 12; x++) hist[Math.min(bins - 1, (g[y * w + x] / max * bins) | 0)]++;
  let b = bins - 1, acc = 0;
  while (b > 0 && acc + hist[b] < count) acc += hist[b--];
  const cut = b / bins * max, pts = [];
  for (let y = 12; y < h - 12; y++) for (let x = 12; x < w - 12; x++) {
    const i = y * w + x;
    if (g[i] >= cut) pts.push([x + 0.5 - w / 2, y + 0.5 - h / 2, g[i]]);
  }
  return pts;
}

// Gaussian blur approximated by three box blurs (running sums, rows then whole-row columns: cache friendly).
function fastBlur(src, w, h, sigma) {
  const r = Math.max(1, Math.round(Math.sqrt(12 * sigma * sigma / 3 + 1) / 2 - 0.5)), d = 2 * r + 1;
  let a = new Float64Array(src), b = new Float64Array(w * h); // float64: running sums in float32 hit subnormal values, which are very slow
  const acc = new Float64Array(w);
  for (let it = 0; it < 3; it++) {
    for (let y = 0; y < h; y++) { // horizontal
      const o = y * w; let s = 0;
      for (let k = -r; k <= r; k++) s += a[o + Math.min(w - 1, Math.max(0, k))];
      for (let x = 0; x < w; x++) {
        b[o + x] = s / d;
        s += a[o + Math.min(w - 1, x + r + 1)] - a[o + Math.max(0, x - r)];
      }
    }
    acc.fill(0); // vertical, a row of running sums at a time
    for (let k = -r; k <= r; k++) { const o = Math.min(h - 1, Math.max(0, k)) * w; for (let x = 0; x < w; x++) acc[x] += b[o + x]; }
    for (let y = 0; y < h; y++) {
      const o = y * w, add = Math.min(h - 1, y + r + 1) * w, sub = Math.max(0, y - r) * w;
      for (let x = 0; x < w; x++) { a[o + x] = acc[x] / d; acc[x] += b[add + x] - b[sub + x]; }
    }
  }
  return a;
}

// Bilinear resample of a frame onto the reference grid (for measuring only); alpha 0 where it has no data.
function warpBilinear(rgba, W, H, a, f) {
  const out = new Uint8ClampedArray(rgba.length), cx = W / 2, cy = H / 2, tx = a.tx * f, ty = a.ty * f;
  for (let y = 0; y < H; y++) {
    const sy = cy + a.s * (y + 0.5 - cy) + ty - 0.5, y0 = Math.floor(sy), fy = sy - y0;
    if (y0 < 0 || y0 > H - 2) continue;
    for (let x = 0; x < W; x++) {
      const sx = cx + a.s * (x + 0.5 - cx) + tx - 0.5, x0 = Math.floor(sx), fx = sx - x0;
      if (x0 < 0 || x0 > W - 2) continue;
      const i = 4 * (y0 * W + x0), j = i + 4 * W, o = 4 * (y * W + x);
      for (let c = 0; c < 3; c++)
        out[o + c] = (rgba[i + c] * (1 - fx) + rgba[i + 4 + c] * fx) * (1 - fy) + (rgba[j + c] * (1 - fx) + rgba[j + 4 + c] * fx) * fy;
      out[o + 3] = 255;
    }
  }
  return out;
}
function alignScore(pts, frm, s, tx, ty) {
  const { w, h } = frm, cx = w / 2, cy = h / 2;
  let sum = 0;
  for (const [dx, dy, wt] of pts) sum += wt * bilinear(frm.g, w, h, cx + s * dx + tx - 0.5, cy + s * dy + ty - 0.5);
  return sum;
}
function searchAlign(pts, frm, start, grid) {
  let best = { ...start, v: alignScore(pts, frm, start.s, start.tx, start.ty) };
  const tryAt = (s, tx, ty) => { const v = alignScore(pts, frm, s, tx, ty); if (v > best.v) best = { s, tx, ty, v }; };
  if (grid.coarse) for (let s = 1 - STACK.scaleRange; s <= 1 + STACK.scaleRange + 1e-9; s += STACK.coarseScale)
    for (let tx = -STACK.maxShift; tx <= STACK.maxShift; tx += STACK.coarseShift)
      for (let ty = -STACK.maxShift; ty <= STACK.maxShift; ty += STACK.coarseShift) tryAt(s, tx, ty);
  for (let round = 0; round < 2; round++) {
    const c = best;
    for (let s = c.s - grid.ds; s <= c.s + grid.ds + 1e-9; s += grid.sStep) tryAt(s, c.tx, c.ty);
    const d = best;
    for (let tx = d.tx - grid.dt; tx <= d.tx + grid.dt + 1e-9; tx += grid.tStep)
      for (let ty = d.ty - grid.dt; ty <= d.ty + grid.dt + 1e-9; ty += grid.tStep) tryAt(d.s, tx, ty);
  }
  return best;
}

function cubicWeights(t, w) {
  const t2 = t * t, t3 = t2 * t;
  w[0] = -0.5 * t3 + t2 - 0.5 * t; w[1] = 1.5 * t3 - 2.5 * t2 + 1; w[2] = -1.5 * t3 + 2 * t2 + 0.5 * t; w[3] = 0.5 * t3 - 0.5 * t2;
}
// Per-channel tone curve mapping img's histogram onto ref's (pixels with alpha 0 left out).
function toneLut(img, ref) {
  const luts = [];
  for (let c = 0; c < 3; c++) {
    const hs = new Float64Array(256), hd = new Float64Array(256);
    for (let j = 0; j < img.length; j += 4) if (img[j + 3] && ref[j + 3]) { hs[img[j + c]]++; hd[ref[j + c]]++; }
    for (let k = 1; k < 256; k++) { hs[k] += hs[k - 1]; hd[k] += hd[k - 1]; }
    const lut = new Uint8Array(256);
    for (let k = 0, d = 0; k < 256; k++) {
      const q = hs[k] / hs[255];
      while (d < 255 && hd[d] / hd[255] < q) d++;
      lut[k] = d;
    }
    luts.push(lut);
  }
  return luts;
}

function boxDilate(m, w, h, r) { // square dilation: counts of set pixels in sliding windows, rows then columns
  const t = new Uint8Array(w * h), o = new Uint8Array(w * h), cnt = new Int32Array(w);
  for (let y = 0; y < h; y++) {
    const row = y * w; let c = 0;
    for (let k = 0; k <= Math.min(w - 1, r); k++) c += m[row + k];
    for (let x = 0; x < w; x++) {
      t[row + x] = c > 0 ? 1 : 0;
      if (x + r + 1 < w) c += m[row + x + r + 1];
      if (x - r >= 0) c -= m[row + x - r];
    }
  }
  for (let k = 0; k <= Math.min(h - 1, r); k++) for (let x = 0; x < w; x++) cnt[x] += t[k * w + x];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) o[y * w + x] = cnt[x] > 0 ? 1 : 0;
    if (y + r + 1 < h) for (let x = 0; x < w; x++) cnt[x] += t[(y + r + 1) * w + x];
    if (y - r >= 0) for (let x = 0; x < w; x++) cnt[x] -= t[(y - r) * w + x];
  }
  return o;
}
function medianLabel(lab, w, h, r, n) { // median of small integer labels in a (2r+1)^2 window, sliding histogram per row
  const out = new Uint8Array(w * h), hist = new Int32Array(n);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
    hist.fill(0); let cnt = 0;
    for (let yy = y0; yy <= y1; yy++) for (let xx = 0; xx <= Math.min(w - 1, r); xx++) { hist[lab[yy * w + xx]]++; cnt++; }
    for (let x = 0; x < w; x++) {
      let k = 0, acc = 0; while (acc + hist[k] <= cnt >> 1) acc += hist[k++];
      out[y * w + x] = k;
      const add = x + r + 1, sub = x - r;
      if (add < w) for (let yy = y0; yy <= y1; yy++) { hist[lab[yy * w + add]]++; cnt++; }
      if (sub >= 0) for (let yy = y0; yy <= y1; yy++) { hist[lab[yy * w + sub]]--; cnt--; }
    }
  }
  return out;
}

async function stackMerge(frames, progress) {
  const S = STACK, n = frames.length, ref = n >> 1;
  const full = await decodeBlob(frames[ref].blob), W = full.w, H = full.h;
  const qw = Math.round(W / 4), qh = Math.round(H / 4), hw = Math.round(W / 2), hh = Math.round(H / 2);
  // 1. alignment
  const refQ = await decodeBlob(frames[ref].blob, qw, qh), refH = await decodeBlob(frames[ref].blob, hw, hh);
  const ptsQ = edgePoints(gradientOf(refQ.rgba, qw, qh), S.samples), ptsH = edgePoints(gradientOf(refH.rgba, hw, hh), 2 * S.samples);
  const al = [];
  for (let i = 0; i < n; i++) {
    progress(`Aligning frame ${i + 1} of ${n}…`);
    if (i === ref) { al.push({ s: 1, tx: 0, ty: 0 }); continue; }
    const q = searchAlign(ptsQ, gradientOf((await decodeBlob(frames[i].blob, qw, qh)).rgba, qw, qh), { s: 1, tx: 0, ty: 0 },
      { coarse: true, ds: S.coarseScale, sStep: S.fineScale, dt: S.coarseShift, tStep: S.fineShift });
    const hres = searchAlign(ptsH, gradientOf((await decodeBlob(frames[i].blob, hw, hh)).rgba, hw, hh), { s: q.s, tx: 2 * q.tx, ty: 2 * q.ty },
      { coarse: false, ds: S.fineScale * 2, sStep: S.halfScale, dt: 1, tStep: S.halfShift });
    al.push({ s: hres.s, tx: hres.tx / 2, ty: hres.ty / 2 });
  }
  // scale against focus is a straight line: frames far off it (blurry ones fool the search) take the fit
  const fv = frames.map(f => f.focus);
  if (n >= 4 && fv.every(v => v != null)) {
    let ok = al.map(() => true), fit = null;
    for (let round = 0; round < 2; round++) {
      const idx = ok.map((v, i) => v ? i : -1).filter(i => i >= 0), m = idx.length;
      const mx = idx.reduce((a, i) => a + fv[i], 0) / m, my = idx.reduce((a, i) => a + al[i].s, 0) / m;
      const sxx = idx.reduce((a, i) => a + (fv[i] - mx) ** 2, 0) || 1, sxy = idx.reduce((a, i) => a + (fv[i] - mx) * (al[i].s - my), 0);
      fit = v => my + sxy / sxx * (v - mx);
      ok = al.map((a, i) => Math.abs(a.s - fit(fv[i])) <= S.lineTol);
    }
    al.forEach((a, i) => { if (i !== ref && Math.abs(a.s - fit(fv[i])) > S.lineTol) { a.s = fit(fv[i]); a.fixed = true; } });
  }
  // 2. tones and sharpness at half size
  const refHalf = refH.rgba, nh = hw * hh, luts = [], meanE = [];
  let bestE = null, label = new Uint8Array(nh);
  for (let i = 0; i < n; i++) {
    progress(`Measuring sharpness ${i + 1} of ${n}…`);
    const half = i === ref ? refHalf : warpBilinear((await decodeBlob(frames[i].blob, hw, hh)).rgba, hw, hh, al[i], 2);
    const lut = i === ref ? null : toneLut(half, refHalf);
    luts.push(lut);
    if (lut) for (let j = 0; j < half.length; j += 4) { half[j] = lut[0][half[j]]; half[j + 1] = lut[1][half[j + 1]]; half[j + 2] = lut[2][half[j + 2]]; }
    const g = lumaOf(half, nh), lap2 = new Float64Array(nh);
    for (let y = 1; y < hh - 1; y++) for (let x = 1; x < hw - 1; x++) {
      const k = y * hw + x, v = 4 * g[k] - g[k - 1] - g[k + 1] - g[k - hw] - g[k + hw]; lap2[k] = half[4 * k + 3] ? v * v : 0;
    }
    const e = fastBlur(lap2, hw, hh, S.energySigma);
    let sum = 0;
    if (!bestE) { bestE = e; for (let k = 0; k < nh; k++) sum += e[k]; }
    else for (let k = 0; k < nh; k++) { sum += e[k]; if (e[k] > bestE[k]) { bestE[k] = e[k]; label[k] = i; } }
    meanE.push(sum / nh);
  }
  // 3. depth surface
  progress('Building the depth map…');
  const sorted = Float64Array.from(bestE.filter((_, k) => k % 7 === 0)).sort(), noise = sorted[Math.floor(sorted.length * 0.2)] || 1e-9;
  const conf = new Float64Array(nh), strong = new Uint8Array(nh);
  for (let k = 0; k < nh; k++) { conf[k] = bestE[k] > S.detail * noise ? 1 : 0; strong[k] = bestE[k] > S.strong * noise ? 1 : 0; }
  const lab = medianLabel(label, hw, hh, S.median, n), labConf = new Float64Array(nh);
  for (let k = 0; k < nh; k++) labConf[k] = lab[k] * conf[k];
  const num = fastBlur(labConf, hw, hh, S.energySigma), den = fastBlur(conf, hw, hh, S.energySigma);
  // wide fallback on an eighth-size grid
  const ew = Math.ceil(hw / 4), eh = Math.ceil(hh / 4), wn = new Float64Array(ew * eh), wd = new Float64Array(ew * eh);
  for (let y = 0; y < hh; y++) for (let x = 0; x < hw; x++) { const e = ((y >> 2) * ew + (x >> 2)), k = y * hw + x; wn[e] += labConf[k]; wd[e] += conf[k]; }
  const wnS = fastBlur(wn, ew, eh, S.wideSigma), wdS = fastBlur(wd, ew, eh, S.wideSigma);
  const depth = new Float32Array(nh);
  for (let y = 0; y < hh; y++) for (let x = 0; x < hw; x++) {
    const k = y * hw + x, e = (y >> 2) * ew + (x >> 2);
    depth[k] = den[k] > 0.05 ? num[k] / den[k] : wdS[e] > 1e-6 ? wnS[e] / wdS[e] : (n - 1) / 2;
  }
  const keepM = boxDilate(strong, hw, hh, S.keepDilate), keepF = new Float64Array(nh);
  for (let k = 0; k < nh; k++) keepF[k] = keepM[k];
  const keep = fastBlur(keepF, hw, hh, S.keepSigma);
  const bg = meanE.indexOf(Math.max(...meanE)); // flat areas: the sharpest frame overall
  // 4. compose at full size: each pixel blends the frames next to its depth
  const nf = W * H, acc = new Float32Array(3 * nf), wsum = new Float32Array(nf), bgRGB = new Uint8ClampedArray(3 * nf);
  // only frames within `reach` of a pixel's depth (weight (1 - dist)^power is negligible beyond), and the background
  // frame only where the stacked result is not used in full
  const dFull = new Float32Array(nf), kFull = new Float32Array(nf), xs0 = new Int32Array(W), xsf = new Float32Array(W);
  for (let x = 0; x < W; x++) { const u = Math.min(hw - 1.001, Math.max(0, x / 2 - 0.25)); xs0[x] = u | 0; xsf[x] = u - (u | 0); }
  for (let y = 0; y < H; y++) {
    const v = Math.min(hh - 1.001, Math.max(0, y / 2 - 0.25)), r0 = (v | 0) * hw, r1 = r0 + hw, fy = v - (v | 0);
    for (let x = 0; x < W; x++) {
      const i = r0 + xs0[x], j = r1 + xs0[x], fx = xsf[x], k = y * W + x;
      dFull[k] = (depth[i] * (1 - fx) + depth[i + 1] * fx) * (1 - fy) + (depth[j] * (1 - fx) + depth[j + 1] * fx) * fy;
      kFull[k] = (keep[i] * (1 - fx) + keep[i + 1] * fx) * (1 - fy) + (keep[j] * (1 - fx) + keep[j + 1] * fx) * fy;
    }
  }
  const cx = W / 2, cy = H / 2, wx = new Float64Array(4), wy = new Float64Array(4), id = new Uint8Array(256).map((_, k) => k);
  for (let i = 0; i < n; i++) {
    progress(`Composing ${i + 1} of ${n}…`);
    const img = i === ref ? full.rgba : (await decodeBlob(frames[i].blob)).rgba, a = al[i];
    const [l0, l1, l2] = luts[i] || [id, id, id], tx = a.tx * 4, ty = a.ty * 4, isBgFrame = i === bg;
    for (let y = 0; y < H; y++) {
      const sy = cy + a.s * (y + 0.5 - cy) + ty - 0.5, y0 = Math.floor(sy);
      if (y0 < 1 || y0 > H - 3) continue;
      cubicWeights(sy - y0, wy);
      const r0 = (y0 - 1) * W;
      for (let x = 0; x < W; x++) {
        const k = y * W + x, dist = Math.abs(dFull[k] - i), use = dist < S.reach, isBg = isBgFrame && kFull[k] < 0.999;
        if (!use && !isBg) continue;
        const sx = cx + a.s * (x + 0.5 - cx) + tx - 0.5, x0 = Math.floor(sx);
        if (x0 < 1 || x0 > W - 3) continue;
        cubicWeights(sx - x0, wx);
        let p0 = 0, p1 = 0, p2 = 0;
        for (let j = 0; j < 4; j++) {
          const row = 4 * (r0 + j * W + x0 - 1), wj = wy[j];
          p0 += wj * (wx[0] * img[row] + wx[1] * img[row + 4] + wx[2] * img[row + 8] + wx[3] * img[row + 12]);
          p1 += wj * (wx[0] * img[row + 1] + wx[1] * img[row + 5] + wx[2] * img[row + 9] + wx[3] * img[row + 13]);
          p2 += wj * (wx[0] * img[row + 2] + wx[1] * img[row + 6] + wx[2] * img[row + 10] + wx[3] * img[row + 14]);
        }
        p0 = l0[p0 < 0 ? 0 : p0 > 255 ? 255 : p0 + 0.5 | 0]; p1 = l1[p1 < 0 ? 0 : p1 > 255 ? 255 : p1 + 0.5 | 0]; p2 = l2[p2 < 0 ? 0 : p2 > 255 ? 255 : p2 + 0.5 | 0];
        if (isBg) { bgRGB[3 * k] = p0; bgRGB[3 * k + 1] = p1; bgRGB[3 * k + 2] = p2; }
        if (use) {
          const wt = (1 - dist) ** S.power;
          acc[3 * k] += wt * p0; acc[3 * k + 1] += wt * p1; acc[3 * k + 2] += wt * p2; wsum[k] += wt;
        }
      }
    }
  }
  const out = new Uint8ClampedArray(4 * nf);
  let detailShare = 0;
  for (let k = 0; k < nh; k++) detailShare += conf[k];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const k = y * W + x, o = 4 * k, kp = wsum[k] > 0 ? Math.min(1, kFull[k]) : 0;
    for (let c = 0; c < 3; c++) out[o + c] = kp * (wsum[k] > 0 ? acc[3 * k + c] / wsum[k] : 0) + (1 - kp) * bgRGB[3 * k + c];
    out[o + 3] = 255;
  }
  return { rgba: out, width: W, height: H, frames: n, background: bg, detail: detailShare / nh,
    shifts: al.map(a => ({ scale: +a.s.toFixed(4), dx: +(a.tx * 4).toFixed(1), dy: +(a.ty * 4).toFixed(1), fixed: !!a.fixed })) };
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
      stackFrames.push({ blob: m.blob, focus: m.focus });
      postMessage({ seq: m.seq, frames: stackFrames.length });
    } else if (m.type === 'stackEnd') {
      const frames = stackFrames; stackFrames = [];
      if (!frames.length) { postMessage({ seq: m.seq, error: 'nothing stacked' }); return; }
      stackMerge(frames, text => postMessage({ progress: text }))
        .then(r => postMessage({ seq: m.seq, ...r, rgba: r.rgba.buffer }, [r.rgba.buffer]))
        .catch(err => postMessage({ seq: m.seq, error: String((err && err.message) || err) }));
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
