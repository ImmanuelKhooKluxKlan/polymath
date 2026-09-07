function positiveInteger(value, fallback = 1, maximum = 32) {
  const parsed = Math.floor(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(1, Math.min(maximum, parsed));
}

class TaskPool {
  constructor(concurrency = 1) {
    this.concurrency = positiveInteger(concurrency);
    this.active = 0;
    this.pending = [];
  }

  run(task) {
    if (typeof task !== 'function') return Promise.reject(new TypeError('Task must be a function.'));
    return new Promise((resolve, reject) => {
      this.pending.push({ task, resolve, reject });
      this.drain();
    });
  }

  drain() {
    while (this.active < this.concurrency && this.pending.length) {
      const entry = this.pending.shift();
      this.active += 1;
      Promise.resolve()
        .then(entry.task)
        .then(entry.resolve, entry.reject)
        .finally(() => {
          this.active -= 1;
          this.drain();
        });
    }
  }

  snapshot() {
    return {
      active: this.active,
      pending: this.pending.length,
      concurrency: this.concurrency,
    };
  }
}

function createTaskPool(concurrency) {
  return new TaskPool(concurrency);
}

module.exports = { TaskPool, createTaskPool, positiveInteger };
