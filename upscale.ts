import { TARGET_4K_EDGE } from "./image";

/** Backend address. Empty in the browser (same server); set VITE_API_URL when building the APK. */
const API_BASE = ((import.meta.env.VITE_API_URL as string | undefined) ?? "").replace(/\/$/, "");

/** Largest edge we send to the AI service (keeps jobs reliable and fast). */
export const MAX_AI_INPUT_EDGE = 2048;
/** Real-ESRGAN is a native 4x model; asking for more is just interpolation, so we never do. */
export const MAX_AI_SCALE = 4;

export interface Health {
  aiConfigured: boolean;
  model?: string;
}

export type UpscalePlan =
  | { kind: "native"; reason: string }
  | { kind: "ai"; inputEdge: number; scale: number; expectedLongEdge: number; reachesTarget: boolean };

/** Decide what honest "4K" means for a given photo. */
export function planUpscale(width: number, height: number): UpscalePlan {
  const longest = Math.max(width, height);
  if (longest >= TARGET_4K_EDGE) {
    return { kind: "native", reason: "This photo already has 4K-class resolution, so no upscaling is needed." };
  }
  const inputEdge = Math.min(longest, MAX_AI_INPUT_EDGE);
  const scale = Math.min(MAX_AI_SCALE, Math.max(2, Math.ceil(TARGET_4K_EDGE / inputEdge)));
  const out = inputEdge * scale;
  return { kind: "ai", inputEdge, scale, expectedLongEdge: Math.min(out, TARGET_4K_EDGE), reachesTarget: out >= TARGET_4K_EDGE };
}

export async function getHealth(): Promise<Health> {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    if (!r.ok) throw new Error();
    return (await r.json()) as Health;
  } catch {
    return { aiConfigured: false };
  }
}

interface JobStatus {
  id: string;
  status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
  error?: string;
}

async function readError(r: Response, fallback: string) {
  try {
    const j = (await r.json()) as { error?: string };
    return j.error || fallback;
  } catch {
    return fallback;
  }
}

/**
 * Sends the (already enhanced) image to the backend for real AI super-resolution
 * and returns the upscaled image. The API key never leaves the server.
 */
export async function aiUpscale(
  image: Blob,
  opts: { scale: number; faceRestore: boolean; signal?: AbortSignal; onStatus: (label: string, fraction: number) => void },
): Promise<Blob> {
  opts.onStatus("Uploading photo", 0.1);
  let created: Response;
  try {
    created = await fetch(`${API_BASE}/api/upscale?scale=${opts.scale}&face=${opts.faceRestore ? 1 : 0}`, {
      method: "POST",
      headers: { "Content-Type": image.type || "image/jpeg" },
      body: image,
      signal: opts.signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new Error("Could not reach the Jashu server. Check your connection and that the backend is running.");
  }
  if (!created.ok) throw new Error(await readError(created, "The upscaling service rejected the request."));
  const { id } = (await created.json()) as { id: string };

  const started = Date.now();
  // Poll until the job finishes (max ~4 minutes).
  while (Date.now() - started < 240_000) {
    await sleep(1800, opts.signal);
    const r = await fetch(`${API_BASE}/api/upscale/${id}`, { signal: opts.signal });
    if (!r.ok) throw new Error(await readError(r, "Lost contact with the upscaling service."));
    const job = (await r.json()) as JobStatus;
    if (job.status === "succeeded") {
      opts.onStatus("Downloading 4K result", 0.92);
      const res = await fetch(`${API_BASE}/api/upscale/${id}/result`, { signal: opts.signal });
      if (!res.ok) throw new Error(await readError(res, "Could not download the upscaled image."));
      return await res.blob();
    }
    if (job.status === "failed" || job.status === "canceled") {
      throw new Error(job.error || "AI upscaling failed. Try a smaller photo or turn off face restore.");
    }
    const elapsed = (Date.now() - started) / 1000;
    opts.onStatus(job.status === "starting" ? "AI model is warming up" : "AI is rebuilding detail", Math.min(0.85, 0.15 + elapsed / 90));
  }
  throw new Error("AI upscaling took too long. Please try again.");
}

function sleep(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}
