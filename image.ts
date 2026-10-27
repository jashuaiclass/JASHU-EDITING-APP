import type { FaceBox } from "./enhance";

export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const ACCEPT_ATTR = ".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp";
export const MAX_UPLOAD_MB = 30;
/** Longest edge we run the pixel enhancement at (bigger photos are processed at this size). */
export const MAX_WORKING_EDGE = 4096;
export const TARGET_4K_EDGE = 3840;

export class UserFacingError extends Error {}

export function validateFile(file: File): void {
  const okType = ACCEPTED_TYPES.includes(file.type) || /\.(jpe?g|png|webp)$/i.test(file.name);
  if (!okType) throw new UserFacingError("Unsupported file type. Please choose a JPG, PNG or WebP image.");
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    throw new UserFacingError(`That file is larger than ${MAX_UPLOAD_MB} MB. Please choose a smaller photo.`);
  }
  if (file.size === 0) throw new UserFacingError("That file is empty.");
}

export function fitWithin(w: number, h: number, maxEdge: number) {
  const longest = Math.max(w, h);
  if (longest <= maxEdge) return { width: w, height: h, scale: 1 };
  const scale = maxEdge / longest;
  return { width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)), scale };
}

export function createCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

/** Decode a file/blob honouring EXIF orientation, optionally capped to a max edge. */
export async function decodeToCanvas(blob: Blob, maxEdge = MAX_WORKING_EDGE): Promise<HTMLCanvasElement> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  } catch {
    throw new UserFacingError("This image could not be read. It may be corrupted or in an unsupported format.");
  }
  try {
    const { width, height } = fitWithin(bitmap.width, bitmap.height, maxEdge);
    return resizeSource(bitmap, width, height);
  } finally {
    bitmap.close();
  }
}

export function getImageSize(blob: Blob): Promise<{ width: number; height: number }> {
  return createImageBitmap(blob, { imageOrientation: "from-image" }).then((b) => {
    const size = { width: b.width, height: b.height };
    b.close();
    return size;
  });
}

/** High-quality resize; large reductions are done in halving steps to avoid aliasing. */
export function resizeSource(src: CanvasImageSource & { width: number; height: number }, w: number, h: number) {
  let curW = src.width;
  let curH = src.height;
  let cur: CanvasImageSource = src;
  while (curW / 2 > w && curH / 2 > h) {
    const nw = Math.floor(curW / 2);
    const nh = Math.floor(curH / 2);
    const step = createCanvas(nw, nh);
    const sctx = step.getContext("2d")!;
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(cur, 0, 0, nw, nh);
    cur = step;
    curW = nw;
    curH = nh;
  }
  const out = createCanvas(w, h);
  const ctx = out.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(cur, 0, 0, w, h);
  return out;
}

export function canvasToBlob(canvas: HTMLCanvasElement, type: "image/jpeg" | "image/png", quality = 0.95): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new UserFacingError("Could not create the image file. Try a smaller size."))),
      type,
      quality,
    );
  });
}

interface FaceDetectorLike {
  detect(image: CanvasImageSource): Promise<{ boundingBox: DOMRectReadOnly }[]>;
}

/** Uses the browser's built-in FaceDetector when available (Chromium). Silently returns [] otherwise. */
export async function detectFaces(canvas: HTMLCanvasElement): Promise<FaceBox[]> {
  try {
    const Ctor = (window as unknown as { FaceDetector?: new (o: object) => FaceDetectorLike }).FaceDetector;
    if (!Ctor) return [];
    const fit = fitWithin(canvas.width, canvas.height, 720);
    const small = resizeSource(canvas, fit.width, fit.height);
    const found = await new Ctor({ fastMode: true, maxDetectedFaces: 6 }).detect(small);
    return found.map(({ boundingBox: b }) => ({
      x: b.x / small.width,
      y: b.y / small.height,
      w: b.width / small.width,
      h: b.height / small.height,
    }));
  } catch {
    return [];
  }
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function formatBytes(n: number) {
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
