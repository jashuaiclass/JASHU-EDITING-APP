import type { EnhanceReport, FaceBox } from "./enhance";
import { enhancePixels } from "./enhance";
import type { WorkerRequest, WorkerResponse } from "./enhance.worker";
import { createCanvas } from "./image";

/** Enhance a canvas and return a new canvas. Uses a Web Worker, falling back to the main thread. */
export async function enhanceCanvas(
  source: HTMLCanvasElement,
  opts: { strength: number; faces: FaceBox[]; onProgress: (fraction: number, label: string) => void },
): Promise<{ canvas: HTMLCanvasElement; report: EnhanceReport }> {
  const { width, height } = source;
  const imageData = source.getContext("2d")!.getImageData(0, 0, width, height);

  let result: { buffer: ArrayBuffer; report: EnhanceReport };
  try {
    result = await runInWorker(imageData, opts);
  } catch (err) {
    if (err instanceof WorkerUnavailable) {
      const report = await enhancePixels(
        { data: imageData.data, width, height },
        { strength: opts.strength, faces: opts.faces, onProgress: opts.onProgress },
      );
      result = { buffer: imageData.data.buffer as ArrayBuffer, report };
    } else {
      throw err;
    }
  }

  const out = createCanvas(width, height);
  out.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(result.buffer), width, height), 0, 0);
  return { canvas: out, report: result.report };
}

class WorkerUnavailable extends Error {}

function runInWorker(
  imageData: ImageData,
  opts: { strength: number; faces: FaceBox[]; onProgress: (fraction: number, label: string) => void },
): Promise<{ buffer: ArrayBuffer; report: EnhanceReport }> {
  return new Promise((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL("./enhance.worker.ts", import.meta.url), { type: "module" });
    } catch {
      reject(new WorkerUnavailable());
      return;
    }
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === "progress") opts.onProgress(msg.fraction, msg.label);
      else if (msg.type === "done") {
        worker.terminate();
        resolve({ buffer: msg.buffer, report: msg.report });
      } else {
        worker.terminate();
        reject(new Error(msg.message));
      }
    };
    worker.onerror = () => {
      worker.terminate();
      reject(new WorkerUnavailable());
    };
    const req: WorkerRequest = {
      buffer: imageData.data.buffer as ArrayBuffer,
      width: imageData.width,
      height: imageData.height,
      strength: opts.strength,
      faces: opts.faces,
    };
    worker.postMessage(req, [req.buffer]);
  });
}
