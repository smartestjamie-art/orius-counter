/* Orius counter: a JavaScript port of photobooth_count.py and the notebook's detector
   (orius_blob_detection.py). Same steps and parameters, checked against the Python version
   on the same photos. Runs in a browser (global `Orius`, also inside a worker) and in Node. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Orius = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Counter version, stored with every result so a photo keeps the count it was given. Change it whenever
  // detection, splitting or the debris model (debris_model.js) changes.
  const VERSION = '2026-10-30j';
  const REFERENCE_RADIUS = 699; // outer dish rim in the notebook's photo
  const DEFAULTS = { background_sigma: 12, gray_threshold: 75, contrast_threshold: 32, component_min_area: 55,
    core_threshold: 50, core_min_pixels: 25 };
  // Cluster splitting and debris filter at the reference scale, as in photobooth_count.py.
  const SPLIT = { split_min_area: 220, peak_min_distance: 8, peak_min_height: 2.0, segment_min_area: 40,
    seed_sigma: 1.0, seed_depth: 3 };
  const DEBRIS = { min_length: 70, min_solidity: 0.75, min_brownness: 0.35 };
  // Backlit dishes: the dark dish-wall ring is cut out of the mask (photobooth_count.BACKLIT).
  const BACKLIT = { sectors: 120, search: [0.95, 1.03], lip_scan: [0.80, 0.94, 0.02], lip_min_sectors: 80, wall_ratio: 0.983, min_dip: 50, min_ring_sectors: 0.5, half_width: 4, max_half: 16, keep_dark: 50, keep_off: 25, keep_grow: 10,
    near: 6, open_radius: 2, split: { seed_depth: 5, peak_min_distance: 10, segment_min_area: 60 } };
  // One insect split in two, on front-lit photos (photobooth_count.MERGE).
  const MERGE = { min_solidity: 0.83, max_length: 42, max_width: 18 };
  // Long thin blobs (sticks, also curved) on front-lit photos (photobooth_count.STICK).
  const STICK = { min_length: 60, max_width: 8.3 };
  // Nymphs, counted on request: warm blobs that are not black adults (photobooth_count.NYMPH).
  const NYMPH = { level: 215, sigma: 1.2, local_sigma: 10, min_tan: 30, min_hue: 0.5, max_blue: 160, min_area: 15, max_sat: 0.9,
    clump: 1.1, max_clump: 6, wall: { sectors: 120, step: 4, outer: 0.75, coverage: 0.33, run: 3, filled: 6 },
    backlit: { min_hue: 0.35, wall: null }, min_red: 60, neutral: 15, gap: 2, max_overlap: 0.4, inside: 0.95,
    tan: { max_hue: 0.8, min_red: 140, min_area: 40 }, brown: { hue: 0.05, max_red: 140, min_area: 55 } };

  // Python's round(): halves go to the even neighbour.
  function pyRound(x) {
    const r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  // ---------- image helpers ----------

  // Pillow's "L" conversion.
  function luma(rgb, n) {
    const g = new Uint8Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 3) g[i] = (rgb[j] * 19595 + rgb[j + 1] * 38470 + rgb[j + 2] * 7471 + 0x8000) >> 16;
    return g;
  }

  // Pillow's LANCZOS resize (fixed-point, horizontal pass then vertical, 8-bit between passes).
  function lanczosCoeffs(inSize, outSize) {
    const filter = x => {
      if (x <= -3 || x >= 3) return 0;
      if (x === 0) return 1;
      const px = Math.PI * x;
      return (Math.sin(px) / px) * (Math.sin(px / 3) / (px / 3));
    };
    const scale = inSize / outSize;
    const filterscale = Math.max(1, scale);
    const support = 3 * filterscale;
    const ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2);
    const kk = new Float64Array(outSize * ksize);
    const PREC = 1 << 22;
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      const ss = 1 / filterscale;
      let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      let ww = 0;
      const k = new Float64Array(xmax);
      for (let x = 0; x < xmax; x++) { const w = filter((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
      for (let x = 0; x < xmax; x++) {
        const v = ww !== 0 ? k[x] / ww : k[x];
        kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * PREC) : Math.trunc(0.5 + v * PREC);
      }
      bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
    }
    return { bounds, kk, ksize };
  }

  function clip8(ss) {
    const v = Math.floor(ss / 4194304); // >> 22
    return v < 0 ? 0 : v > 255 ? 255 : v;
  }

  function resizeRGB(rgb, W, H, w, h) {
    const hc = lanczosCoeffs(W, w), vc = lanczosCoeffs(H, h);
    const tmp = new Uint8Array(w * H * 3);
    for (let y = 0; y < H; y++) {
      for (let xx = 0; xx < w; xx++) {
        const xmin = hc.bounds[xx * 2], xmax = hc.bounds[xx * 2 + 1], ko = xx * hc.ksize;
        let r = 2097152, g = 2097152, b = 2097152; // 1 << 21: round half up
        for (let x = 0; x < xmax; x++) {
          const k = hc.kk[ko + x], j = (y * W + xmin + x) * 3;
          r += rgb[j] * k; g += rgb[j + 1] * k; b += rgb[j + 2] * k;
        }
        const o = (y * w + xx) * 3;
        tmp[o] = clip8(r); tmp[o + 1] = clip8(g); tmp[o + 2] = clip8(b);
      }
    }
    const out = new Uint8Array(w * h * 3);
    for (let yy = 0; yy < h; yy++) {
      const ymin = vc.bounds[yy * 2], ymax = vc.bounds[yy * 2 + 1], ko = yy * vc.ksize;
      for (let x = 0; x < w; x++) {
        let r = 2097152, g = 2097152, b = 2097152;
        for (let y = 0; y < ymax; y++) {
          const k = vc.kk[ko + y], j = ((ymin + y) * w + x) * 3;
          r += tmp[j] * k; g += tmp[j + 1] * k; b += tmp[j + 2] * k;
        }
        const o = (yy * w + x) * 3;
        out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b);
      }
    }
    return out;
  }

  // scipy.ndimage.gaussian_filter: truncate 4, mode "reflect", axis 0 then axis 1.
  function gaussianKernel(sigma) {
    const radius = Math.floor(4 * sigma + 0.5);
    const k = new Float64Array(2 * radius + 1);
    let sum = 0;
    for (let i = -radius; i <= radius; i++) { const v = Math.exp(-0.5 / (sigma * sigma) * i * i); k[i + radius] = v; sum += v; }
    for (let i = 0; i < k.length; i++) k[i] /= sum;
    return { k, radius };
  }

  function reflect(i, n) {
    while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
    return i;
  }

  function gaussianFilter(src, w, h, sigma) {
    const { k, radius } = gaussianKernel(sigma);
    const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
    for (let y = 0; y < h; y++) {
      const o = y * w;
      for (let t = -radius; t <= radius; t++) {
        const kv = k[t + radius], so = reflect(y + t, h) * w;
        for (let x = 0; x < w; x++) tmp[o + x] += kv * src[so + x];
      }
    }
    for (let y = 0; y < h; y++) {
      const o = y * w;
      for (let x = 0; x < w; x++) {
        let acc = 0;
        for (let t = -radius; t <= radius; t++) {
          const xi = x + t;
          acc += k[t + radius] * tmp[o + (xi >= 0 && xi < w ? xi : reflect(xi, w))];
        }
        out[o + x] = acc;
      }
    }
    return out;
  }

  // ---------- dish ----------

  // Strongest circle with radius 25-60% of the shorter side. Edges as in skimage's canny
  // (sigma 2, thresholds 5 / 15); circles found by letting each edge pixel vote along its
  // gradient, normalised by circumference like skimage's hough_circle(normalize=True).
  function findDish(gray, W, H) {
    const f = Math.max(1, pyRound(Math.min(W, H) / 420));
    const w = Math.ceil(W / f), h = Math.ceil(H / f);
    const small = new Float64Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) small[y * w + x] = gray[y * f * W + x * f];
    const sm = gaussianFilter(small, w, h, 2);
    const gx = new Float64Array(w * h), gy = new Float64Array(w * h), mag = new Float64Array(w * h);
    const at = (x, y) => sm[reflect(y, h) * w + reflect(x, w)];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      gx[i] = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      gy[i] = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
      mag[i] = Math.hypot(gx[i], gy[i]);
    }
    // non-maximum suppression
    const cand = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x, m = mag[i];
      if (m < 5) continue;
      const ang = Math.atan2(gy[i], gx[i]) * 180 / Math.PI;
      const a = ((ang % 180) + 180) % 180;
      let o1, o2;
      if (a < 22.5 || a >= 157.5) { o1 = i - 1; o2 = i + 1; }
      else if (a < 67.5) { o1 = i - w - 1; o2 = i + w + 1; }
      else if (a < 112.5) { o1 = i - w; o2 = i + w; }
      else { o1 = i - w + 1; o2 = i + w - 1; }
      if (m >= mag[o1] && m >= mag[o2]) cand[i] = m >= 15 ? 2 : 1;
    }
    // hysteresis: keep weak edges 8-connected to a strong one
    const edge = new Uint8Array(w * h), stack = [];
    for (let i = 0; i < w * h; i++) if (cand[i] === 2) { edge[i] = 1; stack.push(i); }
    while (stack.length) {
      const i = stack.pop(), x = i % w, y = (i - x) / w;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (cand[j] && !edge[j]) { edge[j] = 1; stack.push(j); }
      }
    }
    const short = Math.min(w, h);
    const radii = [];
    for (let r = Math.trunc(short * 0.25); r < Math.trunc(short * 0.6); r += 2) radii.push(r);
    const acc = radii.map(() => new Float32Array(w * h));
    for (let i = 0; i < w * h; i++) {
      if (!edge[i]) continue;
      const x = i % w, y = (i - x) / w, m = mag[i], ux = gx[i] / m, uy = gy[i] / m;
      for (let ri = 0; ri < radii.length; ri++) {
        const r = radii[ri];
        for (const sgn of [1, -1]) {
          const cx = Math.round(x + sgn * r * ux), cy = Math.round(y + sgn * r * uy);
          if (cx >= 0 && cy >= 0 && cx < w && cy < h) acc[ri][cy * w + cx] += 1;
        }
      }
    }
    // Each centre's best gradient-vote score over all radii (votes scatter by a pixel or so: 3x3 sum).
    const vote = new Float64Array(w * h).fill(-1), voteR = new Int32Array(w * h);
    for (let ri = 0; ri < radii.length; ri++) {
      const a = acc[ri], norm = 2 * Math.PI * radii[ri];
      for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const v = (a[i] + a[i - 1] + a[i + 1] + a[i - w] + a[i + w] + a[i - w - 1] + a[i - w + 1] + a[i + w - 1] + a[i + w + 1]) / norm;
        if (v > vote[i]) { vote[i] = v; voteR[i] = radii[ri]; }
      }
    }
    // Gradient voting is only a shortlist: the dish's shadow, the light box or a blurred rim can outvote
    // the rim. Take the 8 strongest separate centres and score each the way skimage's hough_circle
    // does, as the share of its Bresenham perimeter pixels that are edges; the best of those wins.
    const peaks = [], taken = new Uint8Array(w * h);
    for (let k = 0; k < 8; k++) {
      let bi = -1;
      for (let i = 0; i < w * h; i++) if (!taken[i] && vote[i] >= 0 && (bi < 0 || vote[i] > vote[bi])) bi = i;
      if (bi < 0) break;
      const x0 = bi % w, y0 = (bi - x0) / w;
      peaks.push([x0, y0, voteR[bi]]);
      for (let y = Math.max(0, y0 - 15); y <= Math.min(h - 1, y0 + 15); y++)
        for (let x = Math.max(0, x0 - 15); x <= Math.min(w - 1, x0 + 15); x++) taken[y * w + x] = 1;
    }
    let score = -1, rx = 0, ry = 0, rr = 0;
    for (const [bx, by, br] of peaks) {
      for (let ri = 0; ri < radii.length; ri++) {
        const r = radii[ri];
        if (Math.abs(r - br) > 0.12 * br) continue;
        const per = circlePerimeter(r);
        for (let cy = by - 12; cy <= by + 12; cy++) for (let cx = bx - 12; cx <= bx + 12; cx++) {
          let hits = 0;
          for (let k = 0; k < per.length; k += 2) {
            const x = cx + per[k + 1], y = cy + per[k];
            if (x >= 0 && y >= 0 && x < w && y < h && edge[y * w + x]) hits++;
          }
          const v = hits / (per.length / 2);
          if (v > score) { score = v; rx = cx; ry = cy; rr = r; }
        }
      }
    }
    return [rx * f, ry * f, rr * f];
  }

  // skimage.draw.circle_perimeter(0, 0, r) (Bresenham), as interleaved [row, col] offsets.
  function circlePerimeter(r) {
    const out = [];
    let x = 0, y = r, d = 3 - 2 * r;
    while (y >= x) {
      for (const [a, b] of [[y, x], [-y, x], [y, -x], [-y, -x], [x, y], [-x, y], [x, -y], [-x, -y]]) out.push(a, b);
      if (d < 0) d += 4 * x + 6;
      else { d += 4 * (x - y) + 10; y--; }
      x++;
    }
    return out;
  }

  // ---------- notebook detector (orius_blob_detection.detect), first stage ----------

  function scaledParameters(cx, cy, r) {
    const s = r / REFERENCE_RADIUS;
    return { center_x: cx, center_y: cy, radius_x: r, radius_y: r,
      background_sigma: DEFAULTS.background_sigma * s,
      gray_threshold: DEFAULTS.gray_threshold, contrast_threshold: DEFAULTS.contrast_threshold,
      component_min_area: pyRound(DEFAULTS.component_min_area * s * s),
      core_threshold: DEFAULTS.core_threshold,
      core_min_pixels: pyRound(DEFAULTS.core_min_pixels * s * s) };
  }

  // 4-connected labels numbered in raster order of their first pixel (like scipy.ndimage.label).
  function label4(mask, w, h) {
    const labels = new Int32Array(w * h), queue = new Int32Array(w * h);
    const info = [null];
    let n = 0;
    for (let i = 0; i < w * h; i++) {
      if (!mask[i] || labels[i]) continue;
      n++;
      let head = 0, tail = 0, area = 0, minr = h, maxr = -1, minc = w, maxc = -1;
      labels[i] = n; queue[tail++] = i;
      while (head < tail) {
        const j = queue[head++], x = j % w, y = (j - x) / w;
        area++;
        if (y < minr) minr = y; if (y > maxr) maxr = y; if (x < minc) minc = x; if (x > maxc) maxc = x;
        if (y > 0 && mask[j - w] && !labels[j - w]) { labels[j - w] = n; queue[tail++] = j - w; }
        if (x > 0 && mask[j - 1] && !labels[j - 1]) { labels[j - 1] = n; queue[tail++] = j - 1; }
        if (x < w - 1 && mask[j + 1] && !labels[j + 1]) { labels[j + 1] = n; queue[tail++] = j + 1; }
        if (y < h - 1 && mask[j + w] && !labels[j + w]) { labels[j + w] = n; queue[tail++] = j + w; }
      }
      info.push({ area, minr, maxr, minc, maxc });
    }
    return { labels, n, info };
  }

  // scipy.ndimage.binary_fill_holes: background 4-connected to the border stays background.
  function fillHoles(mask, w, h) {
    const outside = new Uint8Array(w * h), stack = new Int32Array(w * h);
    let top = 0;
    const seed = i => { if (!mask[i] && !outside[i]) { outside[i] = 1; stack[top++] = i; } };
    for (let x = 0; x < w; x++) { seed(x); seed((h - 1) * w + x); }
    for (let y = 0; y < h; y++) { seed(y * w); seed(y * w + w - 1); }
    while (top) {
      const j = stack[--top], x = j % w, y = (j - x) / w;
      if (y > 0) seed(j - w);
      if (x > 0) seed(j - 1);
      if (x < w - 1) seed(j + 1);
      if (y < h - 1) seed(j + w);
    }
    const filled = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) filled[i] = outside[i] ? 0 : 1;
    return filled;
  }

  function detect(gray, w, h, p) {
    const n = w * h;
    const bg = gaussianFilter(gray, w, h, p.background_sigma);
    const combined = new Uint8Array(n);
    for (let y = 0; y < h; y++) {
      const dy = (y - p.center_y) / p.radius_y;
      for (let x = 0; x < w; x++) {
        const i = y * w + x, dx = (x - p.center_x) / p.radius_x;
        combined[i] = gray[i] < p.gray_threshold && bg[i] - gray[i] > p.contrast_threshold && dx * dx + dy * dy < 1 ? 1 : 0;
      }
    }
    const filled = fillHoles(combined, w, h);
    const { labels, n: nlab, info } = label4(filled, w, h);
    const components = [];
    for (let id = 1; id <= nlab; id++) {
      const c = info[id];
      if (c.area < p.component_min_area) continue;
      const r0 = Math.max(0, c.minr - 2), r1 = Math.min(h, c.maxr + 3), c0 = Math.max(0, c.minc - 2), c1 = Math.min(w, c.maxc + 3);
      let dark = 0;
      for (let y = r0; y < r1; y++) for (let x = c0; x < c1; x++) {
        const i = y * w + x;
        if (labels[i] === id && gray[i] < p.core_threshold) dark++;
      }
      if (dark < p.core_min_pixels) continue;
      components.push({ blob: id, area: c.area, r0, r1, c0, c1 });
    }
    return { labels, components, parameters: p, bg, filled };
  }

  // ---------- backlit dish wall (photobooth_count.wall_ring / remove_wall) ----------

  function medianOf(v) {
    v.sort((a, b) => a - b);
    const k = v.length >> 1;
    return v.length % 2 ? v[k] : (v[k - 1] + v[k]) / 2;
  }

  // Darkest radius per angular sector between search[0]*r and search[1]*r, and whether the sector shows a
  // dark ring there (photobooth_count.wall_line).
  function wallLine(gray, w, h, cx, cy, r, search) {
    const B = BACKLIT, n = B.sectors;
    const lo = Math.ceil(search[0] * r), hi = Math.floor(search[1] * r), K = hi - lo + 1;
    const y0 = Math.max(0, cy - hi - 1), y1 = Math.min(h, cy + hi + 2), x0 = Math.max(0, cx - hi - 1), x1 = Math.min(w, cx + hi + 2);
    const buckets = Array.from({ length: n * K }, () => []);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const k = Math.floor(Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 0.5);
      if (k >= lo && k <= hi) buckets[sectorOf(x, y, cx, cy) * K + k - lo].push(gray[y * w + x]);
    }
    const line = new Int32Array(n), dark = new Uint8Array(n), width = new Float64Array(n), depth = new Float64Array(n);
    for (let a = 0; a < n; a++) {
      const med = [];
      for (let k = 0; k < K; k++) { const b = buckets[a * K + k]; med.push(b.length ? medianOf(b) : 255); }
      const sm = med.map((_, i) => { const part = med.slice(Math.max(0, i - 1), i + 2); return part.reduce((t, v) => t + v, 0) / part.length; });
      let best = 0;
      for (let i = 1; i < K; i++) if (sm[i] < sm[best]) best = i;
      line[a] = lo + best;
      const md = medianOf(sm.slice());
      dark[a] = md - sm[best] > B.min_dip ? 1 : 0;
      // half width of the dark band: the run around the minimum darker than halfway to the sector's median
      const half = sm[best] + 0.5 * (md - sm[best]);
      let i1 = best, i2 = best;
      while (i1 > 0 && sm[i1 - 1] < half) i1--;
      while (i2 < K - 1 && sm[i2 + 1] < half) i2++;
      width[a] = Math.max(best - i1, i2 - best); depth[a] = sm[best];
    }
    return { line, dark, width, depth, box: [y0, y1, x0, x1] };
  }

  function sectorOf(x, y, cx, cy) {
    const n = BACKLIT.sectors;
    return Math.min(Math.floor((Math.atan2(y - cy, x - cx) + Math.PI) / (2 * Math.PI) * n), n - 1);
  }

  // Median of each ring sector's radius with its ring-showing neighbours (5 sectors).
  function smoothLine(line, dark) {
    const n = line.length, smooth = new Float64Array(n);
    for (let a = 0; a < n; a++) {
      if (!dark[a]) continue;
      const near = [];
      for (let k = -2; k <= 2; k++) { const b = (a + k + n) % n; if (dark[b]) near.push(line[b]); }
      smooth[a] = medianOf(near);
    }
    return smooth;
  }

  const countOf = dark => dark.reduce((t, v) => t + v, 0);

  /** Mask of the dark dish-wall ring, or null when the photo is not backlit. */
  function wallRing(gray, w, h, cx, cy, r) {
    const B = BACKLIT;
    const { line, dark, width, depth, box: [y0, y1, x0, x1] } = wallLine(gray, w, h, cx, cy, r, B.search);
    if (countOf(dark) < B.min_ring_sectors * B.sectors) return null;
    const smooth = smoothLine(line, dark), n = line.length;
    // 1: a light line: cut half_width, keep what is darker (insects); 2: a line darker than keep_dark: cut its whole
    // dark run and keep only dark areas reaching off it (photobooth_count.wall_ring)
    const med5 = v => Array.from(v, (_, a) => medianOf([-2, -1, 0, 1, 2].map(k => v[(a + k + n) % n])));
    const wm = med5(width), dm = med5(depth);
    const hw = wm.map((v, a) => dm[a] < B.keep_dark ? Math.min(B.max_half, Math.max(B.half_width, v + 1)) : B.half_width);
    const ring = new Uint8Array(w * h);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const a = sectorOf(x, y, cx, cy);
      if (dark[a] && Math.abs(Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - smooth[a]) <= hw[a]) ring[y * w + x] = dm[a] < B.keep_dark ? 2 : 1;
    }
    return ring;
  }

  /** Circle [cx, cy, r] through the dark dish wall (photobooth_count.wall_fit), or null. */
  function wallFit(gray, w, h, cx, cy, r, scales) {
    const B = BACKLIT, n = B.sectors;
    let best = null;
    for (const q of scales) {
      const found = wallLine(gray, w, h, cx, cy, q * r, B.search);
      if (!best || countOf(found.dark) > countOf(best.dark)) best = found;
    }
    if (countOf(best.dark) < B.lip_min_sectors) return null;
    const smooth = smoothLine(best.line, best.dark);
    // normal equations of x^2 + y^2 + D x + E y + F = 0, solved by Cramer's rule
    const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], v = [0, 0, 0];
    for (let a = 0; a < n; a++) {
      if (!best.dark[a]) continue;
      const t = (a + 0.5) / n * 2 * Math.PI - Math.PI, px = cx + smooth[a] * Math.cos(t), py = cy + smooth[a] * Math.sin(t);
      const row = [px, py, 1], rhs = -(px * px + py * py);
      for (let i = 0; i < 3; i++) { v[i] += row[i] * rhs; for (let j = 0; j < 3; j++) M[i][j] += row[i] * row[j]; }
    }
    const det3 = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const d = det3(M), sol = [0, 1, 2].map(c => det3(M.map((row, i) => row.map((x, j) => (j === c ? v[i] : x)))) / d);
    const fx = -sol[0] / 2, fy = -sol[1] / 2;
    return [fx, fy, Math.sqrt(fx * fx + fy * fy - sol[2])];
  }

  /** A backlit dish's circle rebuilt from its dark wall, or null for front-lit photos (photobooth_count.wall_dish). */
  function wallDish(gray, w, h, cx, cy, r) {
    const B = BACKLIT;
    let scales = [1.0];
    if (!wallRing(gray, w, h, cx, cy, r)) {
      const [start, stop, step] = B.lip_scan;
      scales = Array.from({ length: Math.ceil((stop - start) / step) }, (_, i) => start + i * step);
    }
    let fit = wallFit(gray, w, h, cx, cy, r, scales);
    if (!fit) return null;
    for (let i = 0; i < 2; i++) fit = wallFit(gray, w, h, pyRound(fit[0]), pyRound(fit[1]), fit[2] / B.wall_ratio, [1.0]) || fit;
    return [fit[0], fit[1], fit[2] / B.wall_ratio];
  }

  // Binary erosion / dilation with a symmetric structuring element given as offsets; outside counts as 0.
  // Only pixels inside box [y0, y1) x [x0, x1) are computed; the rest of the output is 0.
  function morph(mask, w, h, offs, erode, box) {
    const out = new Uint8Array(w * h);
    for (let y = box[0]; y < box[1]; y++) for (let x = box[2]; x < box[3]; x++) {
      let v = erode ? 1 : 0;
      for (const [dx, dy] of offs) {
        const X = x + dx, Y = y + dy, on = X >= 0 && Y >= 0 && X < w && Y < h && mask[Y * w + X];
        if (erode && !on) { v = 0; break; }
        if (!erode && on) { v = 1; break; }
      }
      out[y * w + x] = v;
    }
    return out;
  }

  /** Cut the ring out of the detector's mask and relabel, as detect() would. */
  function removeWall(det, ring, gray, w, h) {
    const B = BACKLIT, p = det.parameters, n = w * h;
    // The ring lies inside the dish's bounding square, so the morphology only needs that square plus a margin.
    const [cx, cy] = [p.center_x, p.center_y], s = p.radius_x / REFERENCE_RADIUS, grow = Math.max(1, pyRound(B.keep_grow * s));
    const m = Math.ceil(BACKLIT.search[1] * p.radius_x) + 2 + 2 * B.near + Math.max(B.open_radius, grow);
    const box = [Math.max(0, cy - m), Math.min(h, cy + m + 1), Math.max(0, cx - m), Math.min(w, cx + m + 1)];
    let near = ring;
    const cross = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]];
    for (let k = 0; k < B.near; k++) near = morph(near, w, h, cross, false, box);
    const diskOf = R => { const d = []; for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (Math.sqrt(dx * dx + dy * dy) <= R) d.push([dx, dy]); return d; };
    // dark pixels near the ring; seeds: dark areas off a dark line's band of at least keep_off pixels, grown back
    // into the band by keep_grow pixels through dark pixels (an insect lying across the line, not the line)
    let wide = near;
    for (let k = 0; k < B.near; k++) wide = morph(wide, w, h, cross, false, box);
    const dark = new Uint8Array(n), seedMask = new Uint8Array(n);
    for (let i = 0; i < n; i++) { dark[i] = gray[i] < B.keep_dark && det.filled[i] && wide[i] ? 1 : 0; seedMask[i] = dark[i] && ring[i] !== 2 ? 1 : 0; }
    const { labels: sl, info: si } = label4(seedMask, w, h);
    let keep = new Uint8Array(n);
    for (let i = 0; i < n; i++) if (sl[i] && si[sl[i]].area >= B.keep_off * s * s) keep[i] = 1;
    for (let k = 0; k < grow; k++) { keep = morph(keep, w, h, cross, false, box); for (let i = 0; i < n; i++) keep[i] &= dark[i]; }
    for (let i = 0; i < n; i++) if (dark[i] && ring[i] === 1) keep[i] = 1;
    const filled = new Uint8Array(n);
    for (let i = 0; i < n; i++) filled[i] = det.filled[i] && !(ring[i] && !keep[i]) ? 1 : 0;
    const disk = diskOf(B.open_radius);
    const opened = morph(morph(filled, w, h, disk, true, box), w, h, disk, false, box);
    for (let i = 0; i < n; i++) if (near[i]) filled[i] = opened[i];
    const { labels, n: nlab, info } = label4(filled, w, h);
    const core = new Int32Array(nlab + 1);
    for (let i = 0; i < n; i++) if (labels[i] && gray[i] < p.core_threshold) core[labels[i]]++;
    const components = [];
    for (let id = 1; id <= nlab; id++) {
      const c = info[id];
      if (c.area < p.component_min_area || core[id] < p.core_min_pixels) continue;
      components.push({ blob: id, area: c.area, r0: Math.max(0, c.minr - 2), r1: Math.min(h, c.maxr + 3),
        c0: Math.max(0, c.minc - 2), c1: Math.min(w, c.maxc + 3) });
    }
    return { ...det, labels, components, filled };
  }

  // ---------- cluster splitting (photobooth_count.resplit) ----------

  // scipy.ndimage.distance_transform_edt (exact; Felzenszwalb & Huttenlocher).
  function edt(m, w, h) {
    const INF = 1e20, f = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) f[i] = m[i] ? INF : 0;
    const len = Math.max(w, h), d = new Float64Array(len), v = new Int32Array(len), z = new Float64Array(len + 1), line = new Float64Array(len);
    const pass = (count, size, get, set) => {
      for (let a = 0; a < count; a++) {
        for (let q = 0; q < size; q++) line[q] = get(a, q);
        let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
        for (let q = 1; q < size; q++) {
          let s = ((line[q] + q * q) - (line[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
          while (s <= z[k]) { k--; s = ((line[q] + q * q) - (line[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
          k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
        }
        k = 0;
        for (let q = 0; q < size; q++) {
          while (z[k + 1] < q) k++;
          d[q] = (q - v[k]) * (q - v[k]) + line[v[k]];
        }
        for (let q = 0; q < size; q++) set(a, q, d[q]);
      }
    };
    pass(w, h, (x, y) => f[y * w + x], (x, y, val) => { f[y * w + x] = val; });
    pass(h, w, (y, x) => f[y * w + x], (y, x, val) => { f[y * w + x] = val; });
    for (let i = 0; i < w * h; i++) f[i] = Math.sqrt(f[i]);
    return f;
  }

  // Square maximum filter, mode "nearest".
  function maxFilter(src, w, h, r) {
    const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let m = -Infinity;
      for (let t = Math.max(0, x - r); t <= Math.min(w - 1, x + r); t++) if (src[y * w + t] > m) m = src[y * w + t];
      tmp[y * w + x] = m;
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let m = -Infinity;
      for (let t = Math.max(0, y - r); t <= Math.min(h - 1, y + r); t++) if (tmp[t * w + x] > m) m = tmp[t * w + x];
      out[y * w + x] = m;
    }
    return out;
  }

  // skimage.feature.peak_local_max(image, min_distance, threshold_abs, exclude_border=False)
  function peakLocalMax(img, w, h, minDistance, threshold) {
    const mx = maxFilter(img, w, h, minDistance);
    let trivial = true;
    for (let i = 0; i < w * h; i++) if (img[i] !== mx[i]) { trivial = false; break; }
    if (trivial) return [];
    const cand = [];
    for (let i = 0; i < w * h; i++) if (img[i] === mx[i] && img[i] > threshold) cand.push(i);
    cand.sort((a, b) => img[b] - img[a]); // stable: ties keep raster order
    const rejected = new Uint8Array(cand.length), out = [];
    for (let a = 0; a < cand.length; a++) {
      if (rejected[a]) continue;
      out.push(cand[a]);
      const ax = cand[a] % w, ay = (cand[a] - ax) / w;
      for (let b = a + 1; b < cand.length; b++) {
        const bx = cand[b] % w, by = (cand[b] - bx) / w;
        if (Math.max(Math.abs(ax - bx), Math.abs(ay - by)) < minDistance) rejected[b] = 1;
      }
    }
    return out;
  }

  // skimage.segmentation.watershed, 4-connected, priority by (value, age) as in skimage.
  function watershed(surface, markers, mask, w, h) {
    const out = new Int32Array(w * h);
    const hv = [], ha = [], hi = [];
    const less = (a, b) => hv[a] < hv[b] || (hv[a] === hv[b] && ha[a] < ha[b]);
    const swap = (a, b) => {
      let t = hv[a]; hv[a] = hv[b]; hv[b] = t; t = ha[a]; ha[a] = ha[b]; ha[b] = t; t = hi[a]; hi[a] = hi[b]; hi[b] = t;
    };
    const push = (v, age, i) => {
      hv.push(v); ha.push(age); hi.push(i);
      let c = hv.length - 1;
      while (c > 0) { const pa = (c - 1) >> 1; if (less(c, pa)) { swap(c, pa); c = pa; } else break; }
    };
    const pop = () => {
      const i = hi[0], last = hv.length - 1;
      swap(0, last); hv.pop(); ha.pop(); hi.pop();
      let c = 0;
      for (;;) {
        const l = 2 * c + 1, r = l + 1;
        if (l >= hv.length) break;
        const s = r < hv.length && less(r, l) ? r : l;
        if (less(s, c)) { swap(s, c); c = s; } else break;
      }
      return i;
    };
    for (let i = 0; i < w * h; i++) if (markers[i] && mask[i]) { out[i] = markers[i]; push(surface[i], 0, i); }
    let age = 0;
    while (hv.length) {
      const i = pop(), x = i % w, y = (i - x) / w;
      const nbs = [y > 0 ? i - w : -1, x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y < h - 1 ? i + w : -1];
      for (const j of nbs) {
        if (j < 0 || !mask[j] || out[j]) continue;
        age++;
        out[j] = out[i];
        push(surface[j], age, j);
      }
    }
    return out;
  }

  // skimage.morphology.h_maxima (float image, 8-connected reconstruction by dilation).
  function hMaxima(img, w, h, hval) {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < w * h; i++) { if (img[i] < mn) mn = img[i]; if (img[i] > mx) mx = img[i]; }
    const out = new Uint8Array(w * h);
    if (hval > mx - mn) return out;
    const rec = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) rec[i] = img[i] - hval;
    let changed = true;
    while (changed) {
      changed = false;
      for (const dir of [1, -1]) {
        for (let k = 0; k < w * h; k++) {
          const i = dir === 1 ? k : w * h - 1 - k, x = i % w, y = (i - x) / w;
          let m = rec[i];
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            const v = rec[yy * w + xx];
            if (v > m) m = v;
          }
          if (m > img[i]) m = img[i];
          if (m > rec[i]) { rec[i] = m; changed = true; }
        }
      }
    }
    for (let i = 0; i < w * h; i++) out[i] = img[i] - rec[i] >= hval - 2e-15 * Math.abs(img[i]) ? 1 : 0;
    return out;
  }

  // ---------- trained debris classifier (photobooth_count.segment_features / insect_probability) ----------

  function segmentFeatures(inSeg, c, w, n, gray, W, bg, rgb, p, s, nseg) {
    let cnt = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, gmin = Infinity, gsum = 0, gsq = 0, core = 0;
    let red = 0, blue = 0, sat = 0, contrast = 0;
    for (let i = 0; i < n; i++) {
      if (!inSeg(i)) continue;
      const x = i % w, y = (i - x) / w, gi = (y + c.r0) * W + x + c.c0, g = gray[gi], j = gi * 3;
      cnt++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
      if (g < gmin) gmin = g; gsum += g; gsq += g * g; if (g < 65) core++;
      const R = rgb[j], G = rgb[j + 1], B = rgb[j + 2], mx = Math.max(R, G, B), mn = Math.min(R, G, B);
      red += R; blue += B; sat += (mx - mn) / (mx + 1); contrast += bg[gi] - g;
    }
    const mx = sx / cnt, my = sy / cnt, a = sxx / cnt - mx * mx, d = syy / cnt - my * my, b = sxy / cnt - mx * my;
    const root = Math.sqrt(((a - d) / 2) ** 2 + b * b), hi = (a + d) / 2 + root, lo = (a + d) / 2 - root;
    const mean = gsum / cnt;
    red /= cnt; blue /= cnt;
    return { area: cnt / s / s, core_frac: core / cnt, min_red: gmin, mean_red: mean, std_red: Math.sqrt(Math.max(0, gsq / cnt - mean * mean)),
      brown: (red - blue) / (red + 1), sat: sat / cnt, contrast: contrast / cnt, elong: Math.sqrt((hi + 0.1) / (lo + 0.1)), nseg,
      comp_area: c.area / s / s, radial: Math.hypot(mx + c.c0 - p.center_x, my + c.r0 - p.center_y) / p.radius_x };
  }

  // Mean leaf probability over the forest; splits compare float32 values like scikit-learn.
  function insectProbability(f, model) {
    const x = model.features.map(k => Math.fround(f[k]));
    let total = 0;
    for (const t of model.trees) {
      let node = 0;
      while (t.l[node] !== -1) node = x[t.f[node]] <= t.t[node] ? t.l[node] : t.r[node];
      total += t.p[node];
    }
    return total / model.trees.length;
  }

  function resplit(det, gray, W, s, rgb, model, split) {
    const SPLIT = split, p = det.parameters, labels = det.labels;
    const splitMin = SPLIT.split_min_area * s * s, segmentMin = SPLIT.segment_min_area * s * s;
    const detections = [];
    for (const c of det.components) {
      const w = c.c1 - c.c0, h = c.r1 - c.r0, n = w * h;
      const m = new Uint8Array(n), dark = new Uint8Array(n), g = new Float64Array(n);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x, gi = (y + c.r0) * W + x + c.c0;
        g[i] = gray[gi];
        m[i] = labels[gi] === c.blob ? 1 : 0;
        dark[i] = m[i] && gray[gi] < p.core_threshold ? 1 : 0;
      }
      const distance = edt(m, w, h);
      const negDist = distance.map(v => -v);
      const candidates = [];
      if (c.area >= splitMin) {
        const peaks = peakLocalMax(distance, w, h, Math.max(2, pyRound(SPLIT.peak_min_distance * s)), SPLIT.peak_min_height * s);
        if (peaks.length) {
          const markers = new Int32Array(n);
          peaks.forEach((i, k) => { markers[i] = k + 1; });
          candidates.push({ surface: negDist, markers, count: peaks.length });
        }
        const smooth = gaussianFilter(g, w, h, SPLIT.seed_sigma * s);
        const inv = new Float64Array(n);
        for (let i = 0; i < n; i++) inv[i] = m[i] ? 255 - smooth[i] : 0;
        const hm = hMaxima(inv, w, h, SPLIT.seed_depth);
        for (let i = 0; i < n; i++) hm[i] = hm[i] && m[i];
        const seeds = label4(hm, w, h);
        if (seeds.n) candidates.push({ surface: smooth, markers: seeds.labels, count: seeds.n });
      }
      if (!candidates.length) {
        let best = 0;
        for (let i = 1; i < n; i++) if (distance[i] > distance[best]) best = i;
        const markers = new Int32Array(n); markers[best] = 1;
        candidates.push({ surface: negDist, markers, count: 1 });
      }
      let best = [], bestSeg = null;
      for (const cand of candidates) {
        const seg = watershed(cand.surface, cand.markers, m, w, h);
        const area = new Int32Array(cand.count + 1), core = new Int32Array(cand.count + 1);
        const sx = new Float64Array(cand.count + 1), sy = new Float64Array(cand.count + 1);
        for (let i = 0; i < n; i++) {
          const k = seg[i]; if (!k) continue;
          area[k]++; if (dark[i]) core[k]++;
          const x = i % w; sx[k] += x; sy[k] += (i - x) / w;
        }
        const kept = [];
        for (let k = 1; k <= cand.count; k++) {
          if (area[k] >= segmentMin && core[k] >= p.core_min_pixels) kept.push({ k, x: sx[k] / area[k] + c.c0, y: sy[k] / area[k] + c.r0, area: area[k], core_pixels: core[k] });
        }
        if (kept.length > best.length) { best = kept; bestSeg = seg; }
      }
      if (!best.length) {
        let a = 0, k = 0, sx = 0, sy = 0;
        for (let i = 0; i < n; i++) if (m[i]) { a++; if (dark[i]) k++; const x = i % w; sx += x; sy += (i - x) / w; }
        best = [{ k: 0, x: sx / a + c.c0, y: sy / a + c.r0, area: a, core_pixels: k }];
      }
      for (const d of best) {
        const out = { blob: c.blob, x: d.x, y: d.y, area: d.area, core_pixels: d.core_pixels };
        if (rgb && model) {
          const inSeg = d.k ? i => bestSeg[i] === d.k : i => m[i] === 1;
          out.features = segmentFeatures(inSeg, c, w, n, gray, W, det.bg, rgb, p, s, best.length);
          out.p_insect = insectProbability(out.features, model);
        }
        detections.push(out);
      }
    }
    return detections;
  }

  // ---------- debris (photobooth_count.debris_blobs) ----------

  function cross(o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }

  function convexHull(pts) {
    pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const lower = [], upper = [];
    for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
    for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
    upper.pop(); lower.pop();
    return lower.concat(upper);
  }

  function debrisBlobs(det, rgb, W, s, minBrownness) {
    const found = new Set();
    for (const c of det.components) {
      let n = 0, sx = 0, sy = 0;
      const xs = [], ys = [];
      for (let y = c.r0; y < c.r1; y++) for (let x = c.c0; x < c.c1; x++) {
        if (det.labels[y * W + x] !== c.blob) continue;
        n++; sx += x; sy += y; xs.push(x); ys.push(y);
      }
      const mx = sx / n, my = sy / n;
      let a = 0, b = 0, d = 0;
      for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; a += dx * dx; b += dx * dy; d += dy * dy; }
      a /= n; b /= n; d /= n;
      const major = 4 * Math.sqrt((a + d) / 2 + Math.sqrt(((a - d) / 2) ** 2 + b * b));
      if (major < DEBRIS.min_length * s) continue;
      // solidity: pixels / pixels whose centres lie in the hull of the pixels' edge midpoints (skimage convex_image)
      const corners = [];
      for (let i = 0; i < n; i++) for (const [ox, oy] of [[-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]]) corners.push([xs[i] + ox, ys[i] + oy]);
      const hull = convexHull(corners);
      let inside = 0;
      for (let y = c.r0; y < c.r1; y++) for (let x = c.c0; x < c.c1; x++) {
        let ok = true;
        for (let k = 0; k < hull.length && ok; k++) if (cross(hull[k], hull[(k + 1) % hull.length], [x, y]) < -1e-9) ok = false;
        if (ok) inside++;
      }
      let r = 0, bl = 0;
      for (let i = 0; i < n; i++) { const j = (ys[i] * W + xs[i]) * 3; r += rgb[j]; bl += rgb[j + 2]; }
      r /= n; bl /= n;
      if (n / inside >= DEBRIS.min_solidity && (r - bl) / Math.max(r, 1) >= minBrownness) found.add(c.blob);
    }
    return found;
  }

  /** Centre, axis lengths (skimage regionprops) and solidity of a component, as debrisBlobs measures them. */
  function componentShape(det, c, W) {
    const xs = [], ys = [];
    let sx = 0, sy = 0;
    for (let y = c.r0; y < c.r1; y++) for (let x = c.c0; x < c.c1; x++) {
      if (det.labels[y * W + x] !== c.blob) continue;
      xs.push(x); ys.push(y); sx += x; sy += y;
    }
    const n = xs.length, mx = sx / n, my = sy / n;
    let a = 0, b = 0, d = 0;
    for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; a += dx * dx; b += dx * dy; d += dy * dy; }
    a /= n; b /= n; d /= n;
    const root = Math.sqrt(((a - d) / 2) ** 2 + b * b);
    const corners = [];
    for (let i = 0; i < n; i++) for (const [ox, oy] of [[-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]]) corners.push([xs[i] + ox, ys[i] + oy]);
    const hull = convexHull(corners);
    let inside = 0;
    for (let y = c.r0; y < c.r1; y++) for (let x = c.c0; x < c.c1; x++) {
      let ok = true;
      for (let k = 0; k < hull.length && ok; k++) if (cross(hull[k], hull[(k + 1) % hull.length], [x, y]) < -1e-9) ok = false;
      if (ok) inside++;
    }
    return { n, x: mx, y: my, major: 4 * Math.sqrt((a + d) / 2 + root), minor: 4 * Math.sqrt(Math.max(0, (a + d) / 2 - root)), solidity: n / inside };
  }

  /** photobooth_count.merge_single: compact, insect-sized components with several segments count once. */
  function mergeSingle(det, cands, W, s) {
    const groups = new Map();
    for (const d of cands) { if (!groups.has(d.blob)) groups.set(d.blob, []); groups.get(d.blob).push(d); }
    for (const c of det.components) {
      const g = groups.get(c.blob);
      if (!g || g.length < 2) continue;
      const f = componentShape(det, c, W);
      if (f.solidity >= MERGE.min_solidity && f.major <= MERGE.max_length * s && f.minor <= MERGE.max_width * s) {
        const keep = g.reduce((a, b) => b.area > a.area ? b : a);
        groups.set(c.blob, [{ ...keep, x: f.x, y: f.y, area: f.n, core_pixels: g.reduce((t, d) => t + d.core_pixels, 0) }]);
      }
    }
    return [...groups.values()].flat();
  }

  /** photobooth_count.stick_blobs: width = 4 x mean distance to the edge, length = area / width. */
  function stickBlobs(det, W, H, s) {
    const found = new Set(), pad = 3;
    for (const c of det.components) {
      const x0 = Math.max(0, c.c0 - pad), y0 = Math.max(0, c.r0 - pad), x1 = Math.min(W, c.c1 + pad), y1 = Math.min(H, c.r1 + pad);
      const w = x1 - x0, h = y1 - y0, m = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) m[y * w + x] = det.filled[(y + y0) * W + x + x0];
      const dt = edt(m, w, h);
      let n = 0, sum = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (det.labels[(y + y0) * W + x + x0] === c.blob) { n++; sum += dt[y * w + x]; }
      const width = 4 * sum / n;
      if (width <= STICK.max_width * s && n / Math.max(width, 1e-9) >= STICK.min_length * s) found.add(c.blob);
    }
    return found;
  }

  // ---------- nymphs ----------

  /** photobooth_count.find_nymphs, worked inside the dish's bounding square: [{x, y, area, kind: 'nymph'}]. */
  function findNymphs(rgb, W, H, grayThreshold, cx, cy, r, adults, backlit) {
    const N = { ...NYMPH, ...(backlit ? NYMPH.backlit : {}) }, s = r / REFERENCE_RADIUS;
    // margin: the blurs below see the same pixels as on the whole photo, out to the dish rim
    const m = Math.ceil(r) + 8 + Math.ceil(4 * N.local_sigma * s + 0.5) + 6;
    const x0 = Math.max(0, cx - m), y0 = Math.max(0, cy - m), w = Math.min(W, cx + m + 1) - x0, h = Math.min(H, cy + m + 1) - y0;
    // Scale each channel so the dish background (75th percentile inside 0.9 r) reads N.level.
    const ch = [0, 1, 2].map(c => {
      const a = new Float64Array(w * h), hist = new Float64Array(256);
      let n = 0;
      for (let y = 0; y < h; y++) for (let x = 0, j = ((y + y0) * W + x0) * 3 + c; x < w; x++, j += 3) {
        a[y * w + x] = rgb[j];
        if (Math.hypot(x + x0 - cx, y + y0 - cy) < 0.9 * r) { hist[rgb[j]]++; n++; }
      }
      const k = Math.floor(0.75 * (n - 1)) + 1;
      let level = 0;
      for (let acc = 0; level < 256; level++) { acc += hist[level]; if (acc >= k) break; }
      const g = gaussianFilter(a, w, h, N.sigma), f = N.level / Math.max(1, level);
      for (let i = 0; i < g.length; i++) g[i] *= f;
      return g;
    });
    const [R, G, B] = ch, n = w * h, box = [0, h, 0, w];
    let warm = new Float64Array(n);
    for (let i = 0; i < n; i++) warm[i] = R[i] - B[i];
    const blur = gaussianFilter(warm, w, h, N.local_sigma * s);
    for (let i = 0; i < n; i++) warm[i] -= blur[i];
    const cross = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]];
    let mask = new Uint8Array(n), black = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      mask[i] = B[i] < N.max_blue && warm[i] > N.min_tan && R[i] >= N.min_red ? 1 : 0;
      const y = (i / w) | 0, x = i - y * w;
      black[i] = rgb[((y + y0) * W + x + x0) * 3] < grayThreshold && R[i] - B[i] < N.neutral ? 1 : 0;
    }
    mask = morph(morph(mask, w, h, cross, true, box), w, h, cross, false, box);
    // Front-lit: food residue where the wall meets the floor makes a ring of warm pixels; skip it where it runs on.
    let wall = null;
    const Wl = NYMPH.wall, rings = Math.floor(r / Wl.step) + 2;
    if (N.wall) {
      const hits = new Uint8Array(rings * Wl.sectors);
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue;
        const y = (i / w) | 0, dx = i - y * w + x0 - cx, dy = y + y0 - cy, d = Math.hypot(dx, dy);
        if (d < Wl.outer * r || d >= r) continue;
        const sec = Math.floor((Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI) * Wl.sectors) % Wl.sectors;
        hits[Math.floor(d / Wl.step) * Wl.sectors + sec] = 1;
      }
      wall = new Uint8Array(rings * Wl.sectors);
      for (let k = 0; k < rings; k++) {
        let c = 0;
        for (let j = 0; j < Wl.sectors; j++) c += hits[k * Wl.sectors + j];
        if (c / Wl.sectors < Wl.coverage) continue;
        const near = new Uint8Array(Wl.sectors);
        for (let kk = Math.max(0, k - 1); kk <= Math.min(rings - 1, k + 1); kk++) for (let j = 0; j < Wl.sectors; j++) if (hits[kk * Wl.sectors + j]) near[j] = 1;
        for (let j = 0; j < Wl.sectors; j++) {
          let run = 0;
          for (let t = -Wl.run; t <= Wl.run; t++) run += near[((j + t) % Wl.sectors + Wl.sectors) % Wl.sectors];
          if (run < Wl.filled) continue;
          for (let kk = k - 1; kk <= k + 1; kk++) if (kk >= 0 && kk < rings) wall[kk * Wl.sectors + j] = 1;
        }
      }
    }
    const at = d => { const x = pyRound(d.x) - x0, y = pyRound(d.y) - y0; return x >= 0 && y >= 0 && x < w && y < h ? y * w + x : -1; };
    for (const d of adults) { const i = at(d); if (i >= 0) black[i] = 1; }
    for (let k = 0; k < N.gap; k++) black = morph(black, w, h, cross, false, box);
    const { labels, n: nl } = label4(mask, w, h);
    const area = new Float64Array(nl + 1), sr = new Float64Array(nl + 1), sg = new Float64Array(nl + 1), sb = new Float64Array(nl + 1);
    const ov = new Float64Array(nl + 1), sx = new Float64Array(nl + 1), sy = new Float64Array(nl + 1);
    for (let i = 0; i < n; i++) {
      const v = labels[i]; if (!v) continue;
      const y = (i / w) | 0, x = i - y * w;
      area[v]++; sr[v] += R[i]; sg[v] += G[i]; sb[v] += B[i]; ov[v] += black[i]; sx[v] += x; sy[v] += y;
    }
    const onAdult = new Uint8Array(nl + 1);
    for (const d of adults) { const i = at(d); if (i >= 0 && labels[i]) onAdult[labels[i]] = 1; }
    const T = N.tan, Bn = N.brown, keep = new Int32Array(nl + 1), blobs = [];
    for (let v = 1; v <= nl; v++) {
      const a = area[v], mr = sr[v] / a, mg = sg[v] / a, mb = sb[v] / a;
      const hue = (mg - mb) / Math.max(mr - mb, 1e-9), sat = (mr - mb) / Math.max(mr, 1e-9);
      const tan = hue >= N.min_hue && hue <= T.max_hue && mr >= T.min_red && a >= N.min_area * s * s;
      const brown = hue >= Bn.hue && hue <= T.max_hue && mr <= Bn.max_red && a >= Bn.min_area * s * s;
      if (!((tan || brown) && sat <= N.max_sat && ov[v] / a <= N.max_overlap && !onAdult[v])) continue;
      const x = sx[v] / a + x0, y = sy[v] / a + y0;
      const dist = Math.hypot(x - cx, y - cy);
      const ring = Math.floor(dist / Wl.step), sec = Math.floor((Math.atan2(y - cy, x - cx) + Math.PI) / (2 * Math.PI) * Wl.sectors) % Wl.sectors;
      const onWall = wall && ring < rings && wall[ring * Wl.sectors + sec];
      if (dist < N.inside * r && !onWall) { keep[v] = blobs.length + 1; blobs.push({ x, y, a, xx: 0, yy: 0, xy: 0 }); }
    }
    if (!blobs.length) return [];
    for (let i = 0; i < n; i++) {
      const k = keep[labels[i]]; if (!k) continue;
      const b = blobs[k - 1], y = (i / w) | 0, dx = i - y * w + x0 - b.x, dy = y + y0 - b.y;
      b.xx += dx * dx; b.yy += dy * dy; b.xy += dx * dy;
    }
    // one nymph's area: the median of the blobs of at least tan.min_area, so faint specks don't shrink it
    const big = blobs.filter(b => b.a >= T.min_area * s * s).map(b => b.a);
    const single = N.clump * medianOf(big.length ? big : blobs.map(b => b.a)), found = [];
    for (const b of blobs) {
      const k = Math.max(1, pyRound(b.a / single)), d = Math.max(b.a - 1, 1);
      if (k > N.max_clump) continue; // food or egg-mass patch
      // long axis of the pixel covariance (np.cov, ddof 1)
      const a = b.xx / d, c = b.yy / d, o = b.xy / d, l = (a + c) / 2 + Math.sqrt(((a - c) / 2) ** 2 + o * o);
      let ex = o, ey = l - a;
      if (Math.abs(ex) + Math.abs(ey) < 1e-12) { ex = l - c; ey = o; }
      if (Math.abs(ex) + Math.abs(ey) < 1e-12) { ex = 0; ey = 1; }
      const norm = Math.hypot(ex, ey), step = 4 * Math.sqrt(Math.max(l, 0)) / k;
      for (let j = 0; j < k; j++) {
        const t = (j - (k - 1) / 2) * step;
        found.push({ x: b.x + t * ex / norm, y: b.y + t * ey / norm, area: b.a / k, kind: 'nymph' });
      }
    }
    return found;
  }

  // ---------- pipeline ----------

  /** Find the dish and bring the photo to the notebook's scale. rgb: Uint8Array, 3 bytes per pixel. */
  function prepare(rgb, W, H, dish) {
    const scaleTo = (cx, cy, r) => {
      if (r <= 1.25 * REFERENCE_RADIUS) return { img: rgb, w: W, h: H, dish: [cx, cy, r], k: 1 };
      const k = REFERENCE_RADIUS / r, w = pyRound(W * k), h = pyRound(H * k);
      return { img: resizeRGB(rgb, W, H, w, h), w, h, dish: [pyRound(cx * k), pyRound(cy * k), REFERENCE_RADIUS], k };
    };
    let p = scaleTo(...(dish || findDish(luma(rgb, W * H), W, H)));
    if (!dish) {
      // Backlit dishes: take the dish from the dark wall ring rather than the circle found above, which
      // sometimes lands on the outer lip.
      const [cx, cy, r] = p.dish, red = new Float64Array(p.w * p.h);
      for (let i = 0, j = 0; i < red.length; i++, j += 3) red[i] = p.img[j];
      const wall = wallDish(red, p.w, p.h, cx, cy, r);
      if (wall) p = scaleTo(...wall.map(v => pyRound(v / p.k)));
    }
    return { rgb: p.img, width: p.w, height: p.h, originalWidth: W, originalHeight: H, dish: p.dish };
  }


  /** Count insects in a prepared photo. Thresholds default to the notebook's 75 / 50. */
  function count(prep, opts) {
    opts = opts || {};
    const [cx, cy, r] = prep.dish, w = prep.width, h = prep.height, s = r / REFERENCE_RADIUS;
    const p = scaledParameters(cx, cy, r);
    if (opts.gray != null) p.gray_threshold = opts.gray;
    if (opts.core != null) p.core_threshold = opts.core;
    if (opts.contrast != null) p.contrast_threshold = opts.contrast;
    const red = new Float64Array(w * h);
    for (let i = 0, j = 0; i < w * h; i++, j += 3) red[i] = prep.rgb[j];
    let det = detect(red, w, h, p);
    // Backlit photos: cut the dark dish wall out, split on deeper dark cores, and sticks need not look brown.
    const ring = wallRing(red, w, h, cx, cy, r);
    if (ring) det = removeWall(det, ring, red, w, h);
    const model = opts.classifier === false ? null
      : opts.model || (typeof ORIUS_DEBRIS_MODEL !== 'undefined' ? ORIUS_DEBRIS_MODEL : null);
    // candidates: every split segment, with why it was dropped (straw rule first, then the classifier) or null
    let candidates = resplit(det, red, w, s, prep.rgb, model, ring ? { ...SPLIT, ...BACKLIT.split } : SPLIT);
    // Front-lit only: packed backlit insects lose real pairs to it.
    if (!ring) candidates = mergeSingle(det, candidates, w, s);
    const debris = debrisBlobs(det, prep.rgb, w, s, ring ? -Infinity : DEBRIS.min_brownness);
    // Backlit sticks mostly lie against insects, which would go with them.
    if (!ring) for (const b of stickBlobs(det, w, h, s)) debris.add(b);
    // The classifier only runs on front-lit photos: on backlit ones it removed as many insects as false hits.
    for (const d of candidates) d.dropped = debris.has(d.blob) ? 'straw' : model && !ring && d.p_insect < model.threshold ? 'classifier' : null;
    const adults = candidates.filter(d => !d.dropped);
    const nymphs = opts.nymphs ? findNymphs(prep.rgb, w, h, p.gray_threshold, cx, cy, r, adults, !!ring) : [];
    const detections = adults.concat(nymphs);
    return { count: detections.length, adults: adults.length, nymphs: nymphs.length, detections, candidates, parameters: p, components: det.components.length, backlit: !!ring,
      debrisFragments: debris.size, debrisRemoved: candidates.filter(d => d.dropped === 'straw').length,
      classifierRemoved: candidates.filter(d => d.dropped === 'classifier').length };
  }

  return { VERSION, REFERENCE_RADIUS, SPLIT, DEBRIS, BACKLIT, NYMPH, STICK, MERGE, prepare, count, findDish, luma, resizeRGB, gaussianFilter, pyRound };
});
