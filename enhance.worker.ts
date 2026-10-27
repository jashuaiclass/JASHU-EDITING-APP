/// Runs the enhancement engine off the main thread so the UI stays responsive.
import { enhancePixels, type EnhanceReport, type FaceBox } from "./enhance";

export interface WorkerRequest {
  buffer: ArrayBuffer;
  width: number;
  height: number;
  strength: number;
  faces: FaceBox[];
}

export type WorkerResponse =
  | { type: "progress"; fraction: number; label: string }
  | { type: "done"; buffer: ArrayBuffer; report: EnhanceReport }
  | { type: "error"; message: string };

const ctx = self as unknown as {
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
};

ctx.onmessage = async (e) => {
  try {
    const { buffer, width, height, strength, faces } = e.data;
    const data = new Uint8ClampedArray(buffer);
    const report = await enhancePixels(
      { data, width, height },
      {
        strength,
        faces,
        onProgress: (fraction, label) => ctx.postMessage({ type: "progress", fraction, label }),
      },
    );
    ctx.postMessage({ type: "done", buffer: data.buffer as ArrayBuffer, report }, [data.buffer as ArrayBuffer]);
  } catch (err) {
    ctx.postMessage({ type: "error", message: err instanceof Error ? err.message : "Enhancement failed" });
  }
};
