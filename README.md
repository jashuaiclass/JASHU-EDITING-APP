# Jashu Editing App

Turn dark, dull, noisy or soft photos into brighter, sharper, natural-looking pictures – with faces left untouched.

**Stack:** React 18 · TypeScript · Tailwind CSS · Vite · Express (secure backend) · Replicate Real-ESRGAN (AI super-resolution)

## Quick start

```bash
npm install
cp .env.example .env        # add REPLICATE_API_TOKEN to enable AI 4K upscaling
npm run dev                 # web on :5173, API on :8787
```

Open http://localhost:5173. Enhancement works with no API key; only AI 4K upscaling needs one.

Production: `npm run build && npm start` (the Express server serves `dist/` and the API on one port).

> Camera capture needs HTTPS (or `localhost`). Deploy behind HTTPS; on unsupported browsers the app falls back to the device's native camera picker.

## How it works

| Feature | Where | Notes |
|---|---|---|
| Camera / upload / drag-drop | `CameraModal.tsx`, `App.tsx` | Permission is requested only after the user taps *Take a Photo*. Handles denied, no camera, busy camera, insecure context. |
| **Enhance Photo** | `src/lib/enhance.ts` (runs in a Web Worker) | Edge-aware denoise → white balance → auto exposure → shadows/highlights → face lighting → contrast → skin-safe vibrance → halo-limited sharpening. Every step is clamped to a conservative range. |
| Face lighting | `detectFaces` in `image.ts` | Uses the browser `FaceDetector` where available (Chromium). Lifts brightness only inside dark faces; never smooths or reshapes. |
| Before/after | `CompareSlider.tsx` | Drag, touch, or arrow keys. Also a side-by-side view. |
| **AI 4K** | `server/index.ts`, `src/lib/upscale.ts` | Real Real-ESRGAN super-resolution via the backend. |
| Export | `App.tsx` | JPEG (q95) or PNG. |

## About "4K" – what is and isn't claimed

- **Enhance Photo is not upscaling.** It never changes pixel dimensions.
- **4K only comes from a real AI upscale** (or from a photo that is already ≥ 3840 px on its long edge, which is exported as-is).
- The AI model is natively 4×. For a small photo, 4× may land below 3840 px; the app then delivers that genuine result and tells you the real size instead of stretching it.
- Photos above 2048 px are sent to the AI at 2048 px (server reliability); the full-resolution enhanced file is always available as the "Enhancement only" download.
- *Restore faces* (GFPGAN) is **off by default** because it can subtly alter facial features.

## Security

- `REPLICATE_API_TOKEN` is read only by the Node server; it is never bundled or sent to the browser.
- Uploads are limited to 12 MB, content-sniffed (JPEG/PNG/WebP magic bytes), and rate-limited per IP (`UPSCALES_PER_HOUR`, default 15).
- Job ids are validated; result downloads are restricted to Replicate's delivery hosts (no SSRF).
- Set `ALLOWED_ORIGIN` if the frontend is hosted on a different origin.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `REPLICATE_API_TOKEN` | – | Enables AI upscaling |
| `REPLICATE_MODEL` | `nightmareai/real-esrgan` | `owner/name` or `owner/name:version` to pin a version |
| `PORT` | `8787` | API/server port |
| `UPSCALES_PER_HOUR` | `15` | Per-IP limit |
| `ALLOWED_ORIGIN` | – | Comma-separated CORS origins |

## Tuning the look

All strengths live at the bottom of `enhancePixels()` in `src/lib/enhance.ts`. The UI slider scales them 20–100; defaults are deliberately conservative to avoid oversaturation, plastic skin and halos.
