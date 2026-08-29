const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');

const execFileAsync = promisify(execFile);
const SCRIPT_PATH = path.join(__dirname, 'omr', 'run_omr.py');
const DEFAULT_TIMEOUT_MS = 25 * 60 * 1000;

function pythonExecutable(environment = process.env) {
  return String(environment.OMR_PYTHON_BIN || (process.platform === 'win32' ? 'python' : 'python3')).trim();
}

function parseWorkerError(error) {
  const stderr = String(error?.stderr || '').trim();
  for (const line of stderr.split(/\r?\n/).reverse()) {
    try {
      const parsed = JSON.parse(line);
      if (parsed?.error) return String(parsed.error);
    } catch {
      // Continue until a structured final worker line is found.
    }
  }
  if (/No module named ['"](?:cv2|pypdfium2|numpy|PIL)/i.test(stderr)) {
    return 'The local sheet reader is not installed on this server. Install server/omr/requirements.txt.';
  }
  if (error?.killed || error?.signal === 'SIGTERM') {
    return 'The local sheet reader exceeded its processing timeout.';
  }
  return stderr.split(/\r?\n/).filter(Boolean).at(-1)
    || error?.message
    || 'The local sheet reader failed.';
}

async function runLocalOmr({
  sourcePath,
  outputPath,
  filename,
  instrument,
  environment = process.env,
}) {
  if (!fs.existsSync(SCRIPT_PATH)) throw new Error('The local OMR worker is missing from this deployment.');
  const timeout = Math.min(
    60 * 60 * 1000,
    Math.max(60_000, Number(environment.OMR_TIMEOUT_MS || DEFAULT_TIMEOUT_MS)),
  );
  const dpi = Math.min(450, Math.max(180, Number(environment.OMR_RENDER_DPI || 300)));
  const maxPages = Math.min(80, Math.max(1, Number(environment.OMR_MAX_PAGES || 20)));
  try {
    const { stdout } = await execFileAsync(pythonExecutable(environment), [
      SCRIPT_PATH,
      '--input', sourcePath,
      '--output', outputPath,
      '--filename', filename,
      '--instrument', instrument,
      '--dpi', String(dpi),
      '--max-pages', String(maxPages),
    ], {
      cwd: path.dirname(SCRIPT_PATH),
      windowsHide: true,
      timeout,
      maxBuffer: 4 * 1024 * 1024,
      env: environment,
    });
    const result = JSON.parse(fs.readFileSync(outputPath, 'utf8'));
    const summaryLine = String(stdout || '').trim().split(/\r?\n/).filter(Boolean).at(-1);
    const summary = summaryLine ? JSON.parse(summaryLine) : {};
    return { result, summary };
  } catch (error) {
    throw new Error(parseWorkerError(error));
  }
}

function localOmrAvailability(environment = process.env) {
  return {
    enabled: String(environment.OMR_ENABLED || 'true').trim().toLowerCase() !== 'false',
    provider: 'Polymath Local OMR',
    python: pythonExecutable(environment),
    renderDpi: Math.min(450, Math.max(180, Number(environment.OMR_RENDER_DPI || 300))),
    maxPages: Math.min(80, Math.max(1, Number(environment.OMR_MAX_PAGES || 20))),
  };
}

module.exports = {
  localOmrAvailability,
  parseWorkerError,
  runLocalOmr,
};
