// engine.worker.js との受け渡し
export class Engine {
  constructor() {
    this.worker = new Worker(new URL("./engine.worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.nextId = 1;
    this.worker.onmessage = (e) => {
      const { id, kind, data, message } = e.data;
      const job = this.pending.get(id);
      if (!job) return;
      if (kind === "progress") {
        job.onProgress?.(data);
        return;
      }
      this.pending.delete(id);
      if (kind === "result") job.resolve(data);
      else if (kind === "cancelled") job.resolve(null);
      else job.reject(new Error(message));
    };
    this.worker.onerror = (e) => {
      for (const job of this.pending.values()) job.reject(new Error(e.message || "処理中にエラーが発生しました。"));
      this.pending.clear();
    };
  }

  // 中止された場合は null で解決する
  call(type, payload, onProgress, transfer = []) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, type, payload }, transfer);
    });
  }

  cancel() {
    this.worker.postMessage({ type: "cancel" });
  }
}
