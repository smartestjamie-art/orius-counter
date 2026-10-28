/* Orius counter: a JavaScript port of photobooth_count.py and the notebook's detector
   (orius_blob_detection.py). Same steps and parameters, checked against the Python version
   on the same photos. Runs in a browser (global `Orius`, also inside a worker) and in Node. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Orius = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const REFERENCE_RADIUS = 699; // outer dish rim in the notebook's photo
  const DEFAULTS = { background_sigma: 12, gray_threshold: 75, contrast_threshold: 32, component_min_area: 55,
    core_threshold: 50, core_min_pixels: 25 };
  // Cluster splitting and debris filter at the reference scale, as in photobooth_count.py.
  const SPLIT = { split_min_area: 220, peak_min_distance: 8, peak_min_height: 2.0, segment_min_area: 40,
    seed_sigma: 1.0, seed_depth: 3 };
  const DEBRIS = { min_length: 70, min_solidity: 0.75, min_brownness: 0.35 };

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
    let best = -1, bx = 0, by = 0, br = 0;
    for (let ri = 0; ri < radii.length; ri++) {
      const a = acc[ri], norm = 2 * Math.PI * radii[ri];
      for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        // votes scatter by a pixel or so: score each centre by its 3x3 neighbourhood
        const v = (a[i] + a[i - 1] + a[i + 1] + a[i - w] + a[i + w] + a[i - w - 1] + a[i - w + 1] + a[i + w - 1] + a[i + w + 1]) / norm;
        if (v > best) { best = v; bx = x; by = y; br = radii[ri]; }
      }
    }
    // Refine near that estimate the way skimage's hough_circle scores a circle: the share of
    // its Bresenham perimeter pixels that are edges. Gradient voting alone tends to lock onto
    // the dish's outer shadow, a few percent wider than the rim the Python version finds.
    let score = -1, rx = bx, ry = by, rr = br;
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
    return { labels, components, parameters: p };
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

  function resplit(det, gray, W, s) {
    const p = det.parameters, labels = det.labels;
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
      let best = [];
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
          if (area[k] >= segmentMin && core[k] >= p.core_min_pixels) kept.push({ x: sx[k] / area[k] + c.c0, y: sy[k] / area[k] + c.r0, area: area[k], core_pixels: core[k] });
        }
        if (kept.length > best.length) best = kept;
      }
      if (!best.length) {
        let a = 0, k = 0, sx = 0, sy = 0;
        for (let i = 0; i < n; i++) if (m[i]) { a++; if (dark[i]) k++; const x = i % w; sx += x; sy += (i - x) / w; }
        best = [{ x: sx / a + c.c0, y: sy / a + c.r0, area: a, core_pixels: k }];
      }
      for (const d of best) detections.push(Object.assign({ blob: c.blob }, d));
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

  function debrisBlobs(det, rgb, W, s) {
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
      // solidity: pixels / pixels whose centres lie in the hull of all pixel corners (skimage convex_image)
      const corners = [];
      for (let i = 0; i < n; i++) for (const [ox, oy] of [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]) corners.push([xs[i] + ox, ys[i] + oy]);
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
      if (n / inside >= DEBRIS.min_solidity && (r - bl) / Math.max(r, 1) >= DEBRIS.min_brownness) found.add(c.blob);
    }
    return found;
  }

  // ---------- pipeline ----------

  /** Find the dish and bring the photo to the notebook's scale. rgb: Uint8Array, 3 bytes per pixel. */
  function prepare(rgb, W, H, dish) {
    let [cx, cy, r] = dish || findDish(luma(rgb, W * H), W, H);
    let w = W, h = H, img = rgb;
    if (r > 1.25 * REFERENCE_RADIUS) {
      const k = REFERENCE_RADIUS / r;
      w = pyRound(W * k); h = pyRound(H * k);
      img = resizeRGB(rgb, W, H, w, h);
      cx = pyRound(cx * k); cy = pyRound(cy * k); r = REFERENCE_RADIUS;
    }
    return { rgb: img, width: w, height: h, originalWidth: W, originalHeight: H, dish: [cx, cy, r] };
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
    const det = detect(red, w, h, p);
    let detections = resplit(det, red, w, s);
    const debris = debrisBlobs(det, prep.rgb, w, s);
    const before = detections.length;
    detections = detections.filter(d => !debris.has(d.blob));
    return { count: detections.length, detections, parameters: p, components: det.components.length,
      debrisFragments: debris.size, debrisRemoved: before - detections.length };
  }

  return { REFERENCE_RADIUS, SPLIT, DEBRIS, prepare, count, findDish, luma, resizeRGB, gaussianFilter, pyRound };
});
