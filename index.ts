/**
 * Jashu Editing App – backend.
 *
 * Purpose: keep the AI super-resolution API key on the server. The browser sends an
 * (already enhanced) image here; the server forwards it to Replicate, polls the job,
 * and streams the finished image back. Nothing secret ever reaches the frontend.
 */
import "dotenv/config";
import express from "express";
import type { NextFunction, Request, Response } from "express";
import cors from "cors";
import rateLimit from "express-rate-limit";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PORT = Number(process.env.PORT ?? 8787);
const TOKEN = process.env.REPLICATE_API_TOKEN?.trim() ?? "";
const MODEL = (process.env.REPLICATE_MODEL ?? "nightmareai/real-esrgan").trim(); // "owner/name" or "owner/name:version"
const API = "https://api.replicate.com/v1";
const MAX_BYTES = 12 * 1024 * 1024;
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];

const app = express();
app.disable("x-powered-by");
app.set("trust proxy", 1);

app.use((_req, res, next) => {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("Permissions-Policy", "camera=(self), microphone=(), geolocation=()");
  next();
});

if (process.env.ALLOWED_ORIGIN) app.use(cors({ origin: process.env.ALLOWED_ORIGIN.split(",") }));

/* ----------------------------- helpers ----------------------------- */

class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

function sniffImage(buf: Buffer): "image/jpeg" | "image/png" | "image/webp" | null {
  if (buf.length > 12 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return "image/jpeg";
  if (buf.length > 8 && buf.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return "image/png";
  if (buf.length > 12 && buf.toString("ascii", 0, 4) === "RIFF" && buf.toString("ascii", 8, 12) === "WEBP") return "image/webp";
  return null;
}

function requireToken() {
  if (!TOKEN) throw new HttpError(503, "AI upscaling is not configured on this server (missing REPLICATE_API_TOKEN).");
}

async function replicate(pathname: string, init: RequestInit = {}): Promise<unknown> {
  let r: globalThis.Response;
  try {
    r = await fetch(`${API}${pathname}`, {
      ...init,
      headers: { Authorization: `Bearer ${TOKEN}`, ...(init.headers ?? {}) },
    });
  } catch {
    throw new HttpError(502, "Could not reach the AI service. Please try again shortly.");
  }
  if (!r.ok) {
    if (r.status === 401) throw new HttpError(502, "The AI service rejected the server's credentials.");
    if (r.status === 402) throw new HttpError(502, "The AI service account is out of credit.");
    if (r.status === 429) throw new HttpError(429, "The AI service is busy. Please wait a moment and retry.");
    if (r.status === 404) throw new HttpError(404, "Not found.");
    throw new HttpError(502, `The AI service returned an error (${r.status}).`);
  }
  return r.json();
}

let versionCache: string | null = null;
async function resolveVersion(): Promise<string> {
  if (MODEL.includes(":")) return MODEL.split(":")[1];
  if (versionCache) return versionCache;
  const m = (await replicate(`/models/${MODEL}`)) as { latest_version?: { id?: string } };
  const id = m.latest_version?.id;
  if (!id) throw new HttpError(502, "Could not determine the AI model version.");
  versionCache = id;
  return id;
}

/** Upload to Replicate's file store; fall back to a data URI if that fails. */
async function stageImage(buf: Buffer, mime: string): Promise<string> {
  try {
    const form = new FormData();
    form.append("content", new Blob([new Uint8Array(buf)], { type: mime }), `photo.${mime.split("/")[1]}`);
    const f = (await replicate("/files", { method: "POST", body: form })) as { urls?: { get?: string } };
    if (f.urls?.get) return f.urls.get;
  } catch (e) {
    if (e instanceof HttpError && e.status !== 502) throw e;
  }
  return `data:${mime};base64,${buf.toString("base64")}`;
}

interface Prediction {
  id: string;
  status: "starting" | "processing" | "succeeded" | "failed" | "canceled";
  error?: string | null;
  output?: string | string[] | null;
}

const ID_RE = /^[a-z0-9]{8,40}$/i;
const idParam = (req: Request) => {
  const id = String(req.params.id ?? "");
  if (!ID_RE.test(id)) throw new HttpError(400, "Invalid job id.");
  return id;
};

const wrap =
  (fn: (req: Request, res: Response) => Promise<void>) => (req: Request, res: Response, next: NextFunction) => {
    fn(req, res).catch(next);
  };

/* ------------------------------ routes ------------------------------ */

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, aiConfigured: !!TOKEN, model: TOKEN ? MODEL.split(":")[0] : undefined });
});

const createLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  limit: Number(process.env.UPSCALES_PER_HOUR ?? 15),
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many upscale requests. Please try again later." },
});

app.post(
  "/api/upscale",
  createLimiter,
  express.raw({ type: IMAGE_TYPES, limit: MAX_BYTES }),
  wrap(async (req, res) => {
    requireToken();
    const body = req.body as unknown;
    if (!Buffer.isBuffer(body) || body.length === 0) throw new HttpError(415, "Send the photo as a JPEG, PNG or WebP request body.");
    const mime = sniffImage(body);
    if (!mime) throw new HttpError(415, "That file is not a valid JPEG, PNG or WebP image.");

    const scale = Number(req.query.scale);
    if (!Number.isInteger(scale) || scale < 2 || scale > 4) throw new HttpError(400, "scale must be 2, 3 or 4.");
    const faceEnhance = req.query.face === "1";

    const [version, imageRef] = await Promise.all([resolveVersion(), stageImage(body, mime)]);
    const p = (await replicate("/predictions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version, input: { image: imageRef, scale, face_enhance: faceEnhance } }),
    })) as Prediction;
    res.status(202).json({ id: p.id, status: p.status });
  }),
);

app.get(
  "/api/upscale/:id",
  wrap(async (req, res) => {
    requireToken();
    const p = (await replicate(`/predictions/${idParam(req)}`)) as Prediction;
    res.setHeader("Cache-Control", "no-store");
    res.json({
      id: p.id,
      status: p.status,
      error: p.status === "failed" ? "The AI model could not process this photo. Try a smaller photo or turn off face restore." : undefined,
    });
  }),
);

app.get(
  "/api/upscale/:id/result",
  wrap(async (req, res) => {
    requireToken();
    const p = (await replicate(`/predictions/${idParam(req)}`)) as Prediction;
    if (p.status !== "succeeded") throw new HttpError(409, "The result is not ready yet.");
    const out = Array.isArray(p.output) ? p.output[0] : p.output;
    if (!out) throw new HttpError(502, "The AI service returned no image.");

    // Only fetch from Replicate's own delivery hosts (guards against SSRF via a tampered URL).
    const url = new URL(out);
    const hostOk = url.protocol === "https:" && (url.hostname === "replicate.delivery" || url.hostname.endsWith(".replicate.delivery") || url.hostname.endsWith(".replicate.com"));
    if (!hostOk) throw new HttpError(502, "Unexpected result location from the AI service.");

    const img = await fetch(url);
    if (!img.ok) throw new HttpError(502, "Could not download the upscaled image.");
    const buf = Buffer.from(await img.arrayBuffer());
    const mime = sniffImage(buf);
    if (!mime) throw new HttpError(502, "The AI service returned an unexpected file.");
    res.setHeader("Content-Type", mime);
    res.setHeader("Cache-Control", "no-store");
    res.send(buf);
  }),
);

/* --------------------- static frontend (production) --------------------- */

const dist = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../dist");
if (fs.existsSync(dist)) {
  app.use(express.static(dist, { maxAge: "1h", index: false }));
  app.get(/^(?!\/api\/).*/, (_req, res) => res.sendFile(path.join(dist, "index.html")));
}

/* ------------------------------ errors ------------------------------ */

app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  const e = err as { status?: number; type?: string; message?: string };
  if (e instanceof HttpError) {
    res.status(e.status).json({ error: e.message });
  } else if (e.type === "entity.too.large") {
    res.status(413).json({ error: "That image is too large to upscale. Try a smaller photo." });
  } else {
    console.error(err);
    res.status(500).json({ error: "Something went wrong on the server." });
  }
});

app.listen(PORT, () => {
  console.log(`Jashu API listening on http://localhost:${PORT}  (AI upscaling ${TOKEN ? "enabled" : "NOT configured"})`);
});
