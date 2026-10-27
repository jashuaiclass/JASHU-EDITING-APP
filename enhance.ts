/**
 * Jashu enhancement engine.
 *
 * Pure pixel maths on an RGBA buffer – no DOM dependency – so it runs in the
 * browser (on canvas ImageData) and can be unit-tested in Node.
 *
 * This is *classical* enhancement (tone, colour, denoise, sharpen). It does not
 * add resolution or invent detail. Real super-resolution lives in the backend
 * (see server/index.ts) and is reported separately in the UI.
 *
 * Design rules to keep results natural:
 *  - every correction is clamped to a conservative range
 *  - tone changes are applied as gains on RGB so hue is preserved
 *  - skin-like hues get reduced saturation boost
 *  - sharpening is luminance-only, thresholded, and overshoot-limited (no halos)
 *  - denoise is edge-aware and mostly chroma (no plastic skin)
 */

export interface PixelBuffer {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** Face rectangles in normalised 0..1 coordinates (optional, from FaceDetector). */
export interface FaceBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface EnhanceOptions {
  /** 0..100, overall strength. 60 is a natural default. */
  strength: number;
  faces?: FaceBox[];
  onProgress?: (fraction: number, label: string) => void;
}

export interface EnhanceReport {
  meanBefore: number;
  meanAfter: number;
  gamma: number;
  wbGain: [number, number, number];
  steps: string[];
}

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

/* ----------------------------- helpers ----------------------------- */

function luma(d: Uint8ClampedArray, i: number) {
  return (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255;
}

/** Separable box blur on a float plane, radius r, edge-clamped. */
function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  if (r < 1) return src.slice();
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const win = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += src[row + clamp(k, 0, w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc / win;
      acc += src[row + clamp(x + r + 1, 0, w - 1)] - src[row + clamp(x - r, 0, w - 1)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += tmp[clamp(k, 0, h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / win;
      acc += tmp[clamp(y + r + 1, 0, h - 1) * w + x] - tmp[clamp(y - r, 0, h - 1) * w + x];
    }
  }
  return out;
}

/** Approximate gaussian via 3 box passes. */
function gaussBlur(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const r = Math.max(1, Math.round(Math.sqrt((12 * sigma * sigma) / 3 + 1) / 2));
  let a = boxBlur(src, w, h, r);
  a = boxBlur(a, w, h, r);
  return boxBlur(a, w, h, r);
}

/** Blur a plane cheaply by working at reduced size, then upsampling bilinearly. */
function blurLarge(src: Float32Array, w: number, h: number, sigmaFrac: number): Float32Array {
  const target = 160; // work on ~160px longest edge
  const f = Math.max(1, Math.floor(Math.max(w, h) / target));
  const sw = Math.max(2, Math.floor(w / f));
  const sh = Math.max(2, Math.floor(h / f));
  const small = new Float32Array(sw * sh);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      let acc = 0;
      let n = 0;
      for (let dy = 0; dy < f; dy++) {
        for (let dx = 0; dx < f; dx++) {
          const sx = x * f + dx;
          const sy = y * f + dy;
          if (sx < w && sy < h) {
            acc += src[sy * w + sx];
            n++;
          }
        }
      }
      small[y * sw + x] = acc / Math.max(1, n);
    }
  }
  const blurred = gaussBlur(small, sw, sh, Math.max(1.5, sigmaFrac * Math.max(sw, sh)));
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = clamp((y / h) * sh - 0.5, 0, sh - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = clamp((x / w) * sw - 0.5, 0, sw - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const tx = fx - x0;
      const top = blurred[y0 * sw + x0] * (1 - tx) + blurred[y0 * sw + x1] * tx;
      const bot = blurred[y1 * sw + x0] * (1 - tx) + blurred[y1 * sw + x1] * tx;
      out[y * w + x] = top * (1 - ty) + bot * ty;
    }
  }
  return out;
}

function histogram(img: PixelBuffer) {
  const hist = new Uint32Array(256);
  const d = img.data;
  const step = Math.max(1, Math.floor((img.width * img.height) / 400000));
  let n = 0;
  let sum = 0;
  let r = 0, g = 0, b = 0;
  for (let p = 0; p < img.width * img.height; p += step) {
    const i = p * 4;
    const l = Math.round(luma(d, i) * 255);
    hist[l]++;
    sum += l;
    r += d[i]; g += d[i + 1]; b += d[i + 2];
    n++;
  }
  const pct = (q: number) => {
    let acc = 0;
    const target = q * n;
    for (let i = 0; i < 256; i++) {
      acc += hist[i];
      if (acc >= target) return i / 255;
    }
    return 1;
  };
  return { mean: sum / n / 255, p01: pct(0.005), p99: pct(0.995), r: r / n, g: g / n, b: b / n };
}

/* ------------------------------ stages ------------------------------ */

/** Edge-aware denoise: strong on chroma, light on luma, only in flat regions. */
function denoise(img: PixelBuffer, amount: number) {
  if (amount <= 0.01) return;
  const { data: d, width: w, height: h } = img;
  const n = w * h;
  const Y = new Float32Array(n);
  const Cb = new Float32Array(n);
  const Cr = new Float32Array(n);
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    Y[p] = 0.299 * r + 0.587 * g + 0.114 * b;
    Cb[p] = -0.168736 * r - 0.331264 * g + 0.5 * b;
    Cr[p] = 0.5 * r - 0.418688 * g - 0.081312 * b;
  }
  const rad = Math.max(1, Math.round(Math.max(w, h) / 1400));
  const Yb = boxBlur(Y, w, h, rad);
  const Cbb = boxBlur(Cb, w, h, rad * 2);
  const Crb = boxBlur(Cr, w, h, rad * 2);

  // local edge strength (|Y - blurred Y|); small = flat, large = edge/detail
  const noiseSigma = 4 + 10 * amount; // grey-levels treated as noise
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    const diff = Math.abs(Y[p] - Yb[p]);
    const flat = Math.exp(-(diff * diff) / (2 * noiseSigma * noiseSigma)); // 1 in flat, ->0 on edges
    const lumaMix = 0.5 * amount * flat; // light, flat-areas only: keeps skin texture
    const chromaMix = clamp(0.85 * amount * (0.4 + 0.6 * flat), 0, 0.9);
    const y = Y[p] * (1 - lumaMix) + Yb[p] * lumaMix;
    const cb = Cb[p] * (1 - chromaMix) + Cbb[p] * chromaMix;
    const cr = Cr[p] * (1 - chromaMix) + Crb[p] * chromaMix;
    d[i] = clamp(y + 1.402 * cr, 0, 255);
    d[i + 1] = clamp(y - 0.344136 * cb - 0.714136 * cr, 0, 255);
    d[i + 2] = clamp(y + 1.772 * cb, 0, 255);
  }
}

/** Global exposure: pick a gamma that moves mean luma toward a pleasant target. */
function autoExposure(img: PixelBuffer, k: number) {
  const st = histogram(img);
  const target = 0.46;
  let gamma = 1;
  if (st.mean > 0.02 && st.mean < 0.98) {
    // mean^gamma = target  ->  gamma = ln(target)/ln(mean)
    const raw = Math.log(target) / Math.log(st.mean);
    gamma = clamp(raw, 0.55, 1.12);
  }
  gamma = 1 + (gamma - 1) * k;
  // gentle levels stretch, never clipping more than ~0.5% of pixels
  const lo = clamp(st.p01, 0, 0.12) * k;
  const hi = 1 - (1 - clamp(st.p99, 0.82, 1)) * k;
  const span = Math.max(0.5, hi - lo);
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v++) {
    const x = clamp((v / 255 - lo) / span, 0, 1);
    lut[v] = Math.round(Math.pow(x, gamma) * 255);
  }
  // apply as a luminance gain so hue/saturation are not shifted
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const l = Math.round(0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]);
    const nl = lut[l];
    const gain = l > 0 ? nl / l : 1;
    const g = clamp(gain, 0.5, 4);
    if (l === 0) {
      d[i] = d[i + 1] = d[i + 2] = nl;
    } else {
      d[i] = clamp(d[i] * g, 0, 255);
      d[i + 1] = clamp(d[i + 1] * g, 0, 255);
      d[i + 2] = clamp(d[i + 2] * g, 0, 255);
    }
  }
  return { gamma, meanBefore: st.mean };
}

/** Grey-world white balance, strongly limited so warm/cool scenes stay believable. */
function whiteBalance(img: PixelBuffer, k: number): [number, number, number] {
  const st = histogram(img);
  const avg = (st.r + st.g + st.b) / 3;
  const raw: [number, number, number] = [avg / Math.max(1, st.r), avg / Math.max(1, st.g), avg / Math.max(1, st.b)];
  const limit = 0.12 * k;
  const gain = raw.map((g) => 1 + clamp(g - 1, -limit, limit)) as [number, number, number];
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = clamp(d[i] * gain[0], 0, 255);
    d[i + 1] = clamp(d[i + 1] * gain[1], 0, 255);
    d[i + 2] = clamp(d[i + 2] * gain[2], 0, 255);
  }
  return gain;
}

/** Local shadow lift / highlight recovery driven by a heavily blurred luminance map. */
function shadowsHighlights(img: PixelBuffer, shadow: number, highlight: number) {
  const { data: d, width: w, height: h } = img;
  const n = w * h;
  const L = new Float32Array(n);
  for (let p = 0, i = 0; p < n; p++, i += 4) L[p] = luma(d, i);
  const local = blurLarge(L, w, h, 0.045);
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    const ll = local[p];
    const lift = shadow * Math.pow(1 - ll, 2.2) * (1 - L[p] * 0.6); // dark regions, skip already-bright pixels
    const pull = highlight * Math.pow(ll, 3) * L[p];
    const gain = clamp(1 + lift - pull, 0.7, 2.2);
    d[i] = clamp(d[i] * gain, 0, 255);
    d[i + 1] = clamp(d[i + 1] * gain, 0, 255);
    d[i + 2] = clamp(d[i + 2] * gain, 0, 255);
  }
}

/** Soft S-curve contrast around mid grey, as a luminance gain. */
function contrast(img: PixelBuffer, amount: number) {
  const lut = new Float32Array(256);
  for (let v = 0; v < 256; v++) {
    const x = v / 255;
    const s = x + amount * (x - 0.5) * (1 - Math.abs(2 * x - 1)) * 1.6; // 0 and 1 stay fixed
    lut[v] = clamp(s, 0, 1) * 255;
  }
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const l = Math.round(0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]);
    if (l < 1) continue;
    const g = clamp(lut[l] / l, 0.6, 1.8);
    d[i] = clamp(d[i] * g, 0, 255);
    d[i + 1] = clamp(d[i + 1] * g, 0, 255);
    d[i + 2] = clamp(d[i + 2] * g, 0, 255);
  }
}

/** Vibrance: lifts dull colours more than vivid ones, goes easy on skin tones. */
function vibrance(img: PixelBuffer, amount: number) {
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    if (mx === mn) continue;
    const sat = (mx - mn) / mx; // 0..1
    // hue in degrees
    const dlt = mx - mn;
    let hue: number;
    if (mx === r) hue = ((g - b) / dlt + 6) % 6;
    else if (mx === g) hue = (b - r) / dlt + 2;
    else hue = (r - g) / dlt + 4;
    hue *= 60;
    const skin = hue >= 8 && hue <= 50 && sat > 0.15 && sat < 0.7 ? 1 : 0;
    const weight = (1 - sat) * (skin ? 0.3 : 1);
    const boost = 1 + amount * weight;
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    d[i] = clamp(lum + (r - lum) * boost, 0, 255);
    d[i + 1] = clamp(lum + (g - lum) * boost, 0, 255);
    d[i + 2] = clamp(lum + (b - lum) * boost, 0, 255);
  }
}

/** Gentle local lift inside detected faces. Brightness only – never smoothing or reshaping. */
function faceLight(img: PixelBuffer, faces: FaceBox[], k: number) {
  const { data: d, width: w, height: h } = img;
  for (const f of faces) {
    const cx = (f.x + f.w / 2) * w;
    const cy = (f.y + f.h / 2) * h;
    const rx = (f.w / 2) * w * 1.15;
    const ry = (f.h / 2) * h * 1.15;
    const x0 = Math.max(0, Math.floor(cx - rx)), x1 = Math.min(w - 1, Math.ceil(cx + rx));
    const y0 = Math.max(0, Math.floor(cy - ry)), y1 = Math.min(h - 1, Math.ceil(cy + ry));
    // measure current face brightness
    let sum = 0, cnt = 0;
    for (let y = y0; y <= y1; y += 2) {
      for (let x = x0; x <= x1; x += 2) {
        sum += luma(d, (y * w + x) * 4);
        cnt++;
      }
    }
    const mean = sum / Math.max(1, cnt);
    const want = clamp(0.5 - mean, 0, 0.14) * k; // lift only if the face is dark; at most ~+14%
    if (want < 0.005) continue;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const nx = (x - cx) / rx;
        const ny = (y - cy) / ry;
        const dist = Math.sqrt(nx * nx + ny * ny);
        if (dist >= 1) continue;
        const fall = 1 - dist * dist; // smooth feather
        const i = (y * w + x) * 4;
        const gain = 1 + want * 2.2 * fall * (1 - luma(d, i) * 0.7);
        d[i] = clamp(d[i] * gain, 0, 255);
        d[i + 1] = clamp(d[i + 1] * gain, 0, 255);
        d[i + 2] = clamp(d[i + 2] * gain, 0, 255);
      }
    }
  }
}

/** Luminance unsharp mask with threshold and overshoot limiting (halo control). */
function sharpen(img: PixelBuffer, amount: number) {
  if (amount <= 0.01) return;
  const { data: d, width: w, height: h } = img;
  const n = w * h;
  const L = new Float32Array(n);
  for (let p = 0, i = 0; p < n; p++, i += 4) L[p] = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  const sigma = clamp(Math.max(w, h) / 1800, 0.7, 2.2);
  const blur = gaussBlur(L, w, h, sigma);
  const lmin = new Float32Array(n);
  const lmax = new Float32Array(n);
  // local min/max in 3x3 for overshoot clamping
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let mn = 255, mx = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = clamp(y + dy, 0, h - 1) * w;
        for (let dx = -1; dx <= 1; dx++) {
          const v = L[yy + clamp(x + dx, 0, w - 1)];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
      }
      lmin[y * w + x] = mn;
      lmax[y * w + x] = mx;
    }
  }
  const threshold = 2.5; // ignore tiny differences => noise is not amplified
  const strength = 0.9 * amount;
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    let detail = L[p] - blur[p];
    const mag = Math.abs(detail);
    if (mag < threshold) continue;
    detail = Math.sign(detail) * (mag - threshold * 0.5);
    const target = clamp(L[p] + detail * strength, lmin[p] - 6, lmax[p] + 6); // limit overshoot
    const diff = target - L[p];
    d[i] = clamp(d[i] + diff, 0, 255);
    d[i + 1] = clamp(d[i + 1] + diff, 0, 255);
    d[i + 2] = clamp(d[i + 2] + diff, 0, 255);
  }
}

/* ------------------------------- driver ------------------------------- */

export async function enhancePixels(img: PixelBuffer, opts: EnhanceOptions): Promise<EnhanceReport> {
  const k = clamp(opts.strength, 0, 100) / 100;
  const prog = (f: number, label: string) => opts.onProgress?.(f, label);
  const steps: string[] = [];
  const before = histogram(img);

  // Very dark images get a slightly stronger denoise since brightening amplifies noise.
  const darkness = clamp((0.4 - before.mean) / 0.4, 0, 1);

  prog(0.05, "Analysing photo");
  await tick();

  denoise(img, k * (0.45 + 0.55 * darkness));
  steps.push("Noise reduction (edge-aware, chroma-first)");
  prog(0.2, "Reducing noise");
  await tick();

  const wb = whiteBalance(img, k);
  steps.push("White balance");
  prog(0.32, "Balancing colour");
  await tick();

  // Dark photos need a stronger exposure move than the slider alone gives, but still bounded.
  const exp = autoExposure(img, clamp(k + darkness * 0.45, 0, 1));
  steps.push("Exposure & levels");
  prog(0.45, "Correcting exposure");
  await tick();

  shadowsHighlights(img, 0.55 * k * (0.5 + darkness), 0.25 * k);
  steps.push("Shadows & highlights");
  prog(0.6, "Recovering shadows and highlights");
  await tick();

  if (opts.faces && opts.faces.length) {
    faceLight(img, opts.faces, k);
    steps.push(`Face lighting (${opts.faces.length} face${opts.faces.length > 1 ? "s" : ""})`);
  }
  prog(0.7, "Lighting faces");
  await tick();

  contrast(img, 0.16 * k);
  steps.push("Contrast");
  prog(0.78, "Adding depth");
  await tick();

  vibrance(img, 0.35 * k);
  steps.push("Natural vibrance (skin-safe)");
  prog(0.88, "Refining colour");
  await tick();

  sharpen(img, k);
  steps.push("Sharpening (halo-limited)");
  prog(1, "Finishing");
  await tick();

  const after = histogram(img);
  return { meanBefore: before.mean, meanAfter: after.mean, gamma: exp.gamma, wbGain: wb, steps };
}
