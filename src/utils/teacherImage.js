const ALLOWED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const SOURCE_IMAGE_LIMIT_BYTES = 20 * 1024 * 1024;

function loadImageElement(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new window.Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('The selected character image could not be read.'));
    };
    image.src = url;
  });
}

function canvasBlob(canvas, type, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('The browser could not prepare this character image.'));
    }, type, quality);
  });
}

export async function normalizeTeacherImage(file, options = {}) {
  if (!(file instanceof Blob)) throw new Error('Choose a character image first.');
  if (!ALLOWED_IMAGE_TYPES.has(String(file.type || '').toLowerCase())) {
    throw new Error('Choose a PNG, JPEG, or WebP image.');
  }
  if (file.size > SOURCE_IMAGE_LIMIT_BYTES) throw new Error('Choose an image smaller than 20 MB.');

  const width = Math.max(320, Math.min(1600, Number(options.width) || 768));
  const height = Math.max(400, Math.min(2000, Number(options.height) || 960));
  const source = typeof window.createImageBitmap === 'function'
    ? await window.createImageBitmap(file)
    : await loadImageElement(file);
  const sourceWidth = Number(source.width || source.naturalWidth);
  const sourceHeight = Number(source.height || source.naturalHeight);
  if (!sourceWidth || !sourceHeight) throw new Error('The character image has invalid dimensions.');

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { alpha: true });
  if (!context) throw new Error('The browser could not prepare this character image.');
  context.clearRect(0, 0, width, height);
  const safeWidth = width * 0.94;
  const safeHeight = height * 0.94;
  const scale = Math.min(safeWidth / sourceWidth, safeHeight / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const drawX = (width - drawWidth) / 2;
  const drawY = height - drawHeight - height * 0.03;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, drawX, drawY, drawWidth, drawHeight);
  source.close?.();

  let blob;
  try {
    blob = await canvasBlob(canvas, 'image/webp', 0.92);
  } catch {
    blob = await canvasBlob(canvas, 'image/png');
  }
  const extension = blob.type === 'image/png' ? 'png' : 'webp';
  return new window.File([blob], `teacher-${Date.now()}.${extension}`, { type: blob.type });
}
