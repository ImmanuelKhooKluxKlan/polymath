import fs from 'node:fs';
import path from 'node:path';

const remoteOrigin = String(process.env.VITE_ASSET_BASE_URL || '').trim();
if (!remoteOrigin) {
  console.log('VITE_ASSET_BASE_URL is empty; bundled instrument samples are retained.');
  process.exit(0);
}

const buildRoot = path.resolve(process.cwd(), 'dist');
const sampleRoot = path.resolve(buildRoot, 'samples');
const expectedPrefix = `${buildRoot}${path.sep}`;
if (!sampleRoot.startsWith(expectedPrefix)) {
  throw new Error('Refusing to prune a path outside the frontend build directory.');
}

if (fs.existsSync(sampleRoot)) {
  fs.rmSync(sampleRoot, { recursive: true, force: true });
  console.log('Removed bundled samples from the CDN frontend build; R2 will serve them.');
}
