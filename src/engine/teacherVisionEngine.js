import { midiToNoteName } from './teacherHearingEngine.js';

const WHITE_PITCH_CLASSES = new Set([0, 2, 4, 5, 7, 9, 11]);

function clamp(value, minimum = 0, maximum = 1) {
  return Math.max(minimum, Math.min(maximum, value));
}

function isWhiteMidi(midi) {
  return WHITE_PITCH_CLASSES.has(((midi % 12) + 12) % 12);
}

function normalizeFirstWhiteMidi(midi) {
  let candidate = Math.round(Number(midi) || 48);
  candidate = clamp(candidate, 21, 108);
  while (candidate <= 108 && !isWhiteMidi(candidate)) candidate += 1;
  return Math.min(candidate, 108);
}

export const WHITE_KEY_OPTIONS = Object.freeze(
  Array.from({ length: 88 }, (_, index) => index + 21)
    .filter(isWhiteMidi)
    .map((midi) => ({ midi, label: midiToNoteName(midi) })),
);

export function buildKeyboardGeometry({ firstWhiteMidi = 48, whiteKeyCount = 21 } = {}) {
  const requestedCount = clamp(Math.round(Number(whiteKeyCount) || 21), 3, 52);
  const whiteKeys = [];
  let midi = normalizeFirstWhiteMidi(firstWhiteMidi);

  while (midi <= 108 && whiteKeys.length < requestedCount) {
    if (isWhiteMidi(midi)) whiteKeys.push(midi);
    midi += 1;
  }

  const count = whiteKeys.length;
  const whites = whiteKeys.map((keyMidi, index) => ({
    midi: keyMidi,
    note: midiToNoteName(keyMidi),
    type: 'white',
    xStart: index / count,
    xEnd: (index + 1) / count,
    xCenter: (index + 0.5) / count,
    yStart: 0.58,
    yEnd: 0.98,
  }));

  const blacks = [];
  for (let index = 0; index < whiteKeys.length - 1; index += 1) {
    if (whiteKeys[index + 1] - whiteKeys[index] !== 2) continue;
    const width = 0.52 / count;
    const center = (index + 1) / count;
    const keyMidi = whiteKeys[index] + 1;
    blacks.push({
      midi: keyMidi,
      note: midiToNoteName(keyMidi),
      type: 'black',
      xStart: clamp(center - (width / 2)),
      xEnd: clamp(center + (width / 2)),
      xCenter: center,
      yStart: 0.03,
      yEnd: 0.6,
    });
  }

  return { firstWhiteMidi: whiteKeys[0], whiteKeyCount: count, whites, blacks, keys: [...whites, ...blacks] };
}

function grayAt(data, width, x, y) {
  const offset = ((y * width) + x) * 4;
  return (data[offset] * 0.299) + (data[offset + 1] * 0.587) + (data[offset + 2] * 0.114);
}

function scoreBand(imageData, width, height, normalizedY, normalizedHeight) {
  const data = imageData.data || imageData;
  const xStart = Math.max(2, Math.floor(width * 0.04));
  const xEnd = Math.min(width - 2, Math.ceil(width * 0.96));
  const yStart = Math.max(2, Math.floor(height * normalizedY));
  const yEnd = Math.min(height - 2, Math.ceil(height * (normalizedY + normalizedHeight)));
  let sum = 0;
  let squared = 0;
  let verticalEdges = 0;
  let horizontalEdges = 0;
  let upperDark = 0;
  let lowerLight = 0;
  let count = 0;
  let upperCount = 0;
  let lowerCount = 0;

  for (let y = yStart + 2; y < yEnd - 2; y += 2) {
    const upper = y < yStart + ((yEnd - yStart) * 0.58);
    for (let x = xStart + 2; x < xEnd - 2; x += 2) {
      const gray = grayAt(data, width, x, y);
      sum += gray;
      squared += gray * gray;
      verticalEdges += Math.abs(gray - grayAt(data, width, x - 2, y));
      horizontalEdges += Math.abs(gray - grayAt(data, width, x, y - 2));
      count += 1;
      if (upper) {
        upperCount += 1;
        if (gray < 72) upperDark += 1;
      } else {
        lowerCount += 1;
        if (gray > 145) lowerLight += 1;
      }
    }
  }

  if (!count) return { confidence: 0, brightness: 0, contrast: 0, directional: 0 };
  const brightness = sum / count;
  const variance = Math.max(0, (squared / count) - (brightness * brightness));
  const contrast = Math.sqrt(variance);
  const verticalMean = verticalEdges / count;
  const horizontalMean = horizontalEdges / count;
  const directional = clamp((verticalMean - (horizontalMean * 0.68) - 2) / 22);
  const contrastScore = clamp((contrast - 18) / 58);
  const tonalScore = clamp((1 - Math.abs(brightness - 132) / 132));
  const blackWhiteScore = Math.min(
    clamp((upperDark / Math.max(1, upperCount)) / 0.17),
    clamp((lowerLight / Math.max(1, lowerCount)) / 0.42),
  );

  return {
    confidence: clamp((directional * 0.45) + (contrastScore * 0.26) + (blackWhiteScore * 0.21) + (tonalScore * 0.08)),
    brightness,
    contrast,
    directional,
  };
}

export function findKeyboardCandidate(imageData, width, height) {
  if (!imageData || width < 40 || height < 30) {
    return { detected: false, confidence: 0, region: { x: 0.05, y: 0.48, width: 0.9, height: 0.32 } };
  }

  const normalizedHeight = 0.3;
  let best = null;
  for (let y = 0.12; y <= 0.68; y += 0.07) {
    const score = scoreBand(imageData, width, height, y, normalizedHeight);
    if (!best || score.confidence > best.confidence) best = { ...score, y };
  }

  return {
    detected: best.confidence >= 0.34,
    confidence: best.confidence,
    brightness: best.brightness,
    contrast: best.contrast,
    region: { x: 0.04, y: best.y, width: 0.92, height: normalizedHeight },
  };
}

export function extractRegionGray(imageData, frameWidth, frameHeight, region) {
  const data = imageData.data || imageData;
  const xStart = clamp(Math.floor(region.x * frameWidth), 0, frameWidth - 1);
  const yStart = clamp(Math.floor(region.y * frameHeight), 0, frameHeight - 1);
  const width = Math.max(1, Math.min(frameWidth - xStart, Math.floor(region.width * frameWidth)));
  const height = Math.max(1, Math.min(frameHeight - yStart, Math.floor(region.height * frameHeight)));
  const gray = new Uint8Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      gray[(y * width) + x] = Math.round(grayAt(data, frameWidth, xStart + x, yStart + y));
    }
  }
  return { gray, width, height, region: { ...region } };
}

export function regionToCorners(region) {
  return [
    { x: region.x, y: region.y },
    { x: region.x + region.width, y: region.y },
    { x: region.x + region.width, y: region.y + region.height },
    { x: region.x, y: region.y + region.height },
  ];
}

function pointInQuad(points, horizontal, vertical) {
  const top = {
    x: points[0].x + ((points[1].x - points[0].x) * horizontal),
    y: points[0].y + ((points[1].y - points[0].y) * horizontal),
  };
  const bottom = {
    x: points[3].x + ((points[2].x - points[3].x) * horizontal),
    y: points[3].y + ((points[2].y - points[3].y) * horizontal),
  };
  return {
    x: top.x + ((bottom.x - top.x) * vertical),
    y: top.y + ((bottom.y - top.y) * vertical),
  };
}

function bilinearGray(data, width, height, normalizedX, normalizedY) {
  const x = clamp(normalizedX, 0, 1) * (width - 1);
  const y = clamp(normalizedY, 0, 1) * (height - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const horizontal = x - x0;
  const vertical = y - y0;
  const top = grayAt(data, width, x0, y0) * (1 - horizontal) + grayAt(data, width, x1, y0) * horizontal;
  const bottom = grayAt(data, width, x0, y1) * (1 - horizontal) + grayAt(data, width, x1, y1) * horizontal;
  return Math.round(top * (1 - vertical) + bottom * vertical);
}

export function extractPerspectiveGray(
  imageData,
  frameWidth,
  frameHeight,
  points,
  outputWidth = 252,
  outputHeight = 92,
) {
  if (!Array.isArray(points) || points.length !== 4) {
    throw new Error('Four keyboard corner points are required.');
  }
  const data = imageData.data || imageData;
  const width = Math.max(24, Math.round(outputWidth));
  const height = Math.max(16, Math.round(outputHeight));
  const gray = new Uint8Array(width * height);

  for (let y = 0; y < height; y += 1) {
    const vertical = height === 1 ? 0 : y / (height - 1);
    for (let x = 0; x < width; x += 1) {
      const horizontal = width === 1 ? 0 : x / (width - 1);
      const source = pointInQuad(points, horizontal, vertical);
      gray[(y * width) + x] = bilinearGray(data, frameWidth, frameHeight, source.x, source.y);
    }
  }

  return { gray, width, height, points: points.map((point) => ({ ...point })) };
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function zoneMotionScore(baseline, current, width, height, zone, globalShift = 0) {
  const xPadding = (zone.xEnd - zone.xStart) * 0.13;
  const xStart = clamp(Math.floor((zone.xStart + xPadding) * width), 0, width - 1);
  const xEnd = clamp(Math.ceil((zone.xEnd - xPadding) * width), xStart + 1, width);
  const yStart = clamp(Math.floor(zone.yStart * height), 0, height - 1);
  const yEnd = clamp(Math.ceil(zone.yEnd * height), yStart + 1, height);
  let sum = 0;
  let count = 0;
  const stride = width > 260 ? 2 : 1;

  for (let y = yStart; y < yEnd; y += stride) {
    for (let x = xStart; x < xEnd; x += stride) {
      const index = (y * width) + x;
      sum += Math.abs((current[index] - baseline[index]) - globalShift);
      count += 1;
    }
  }
  return count ? sum / count : 0;
}

export function detectKeyMotion({ baseline, current, width, height, geometry, threshold } = {}) {
  if (!baseline || !current || baseline.length !== current.length || baseline.length !== width * height) {
    return { pressed: [], scores: [], threshold: Number.POSITIVE_INFINITY, noiseFloor: 0 };
  }

  let shiftSum = 0;
  let shiftCount = 0;
  for (let index = 0; index < current.length; index += 7) {
    shiftSum += current[index] - baseline[index];
    shiftCount += 1;
  }
  const globalShift = shiftCount ? shiftSum / shiftCount : 0;
  const scores = geometry.keys.map((key) => ({
    ...key,
    score: zoneMotionScore(baseline, current, width, height, key, globalShift),
  }));
  const noiseFloor = median(scores.map((entry) => entry.score));
  const activationThreshold = Number.isFinite(threshold)
    ? threshold
    : Math.max(8, (noiseFloor * 2.15) + 4.5);
  const candidates = scores
    .filter((entry) => entry.score >= activationThreshold)
    .sort((left, right) => right.score - left.score);
  const pressed = [];

  for (const candidate of candidates) {
    const candidateWidth = candidate.xEnd - candidate.xStart;
    const overlapsStronger = pressed.some((selected) => (
      Math.abs(selected.xCenter - candidate.xCenter) < Math.max(candidateWidth, selected.xEnd - selected.xStart) * 0.46
    ));
    if (overlapsStronger) continue;
    pressed.push({
      ...candidate,
      confidence: clamp((candidate.score - activationThreshold) / 32),
    });
    if (pressed.length >= 5) break;
  }

  return { pressed, scores, threshold: activationThreshold, noiseFloor, globalShift };
}

export function drawTeacherVisionOverlay(context, {
  width,
  height,
  region,
  points,
  manualPoints = [],
  geometry,
  pressed = [],
  calibrated = false,
  detected = false,
  confidence = 0,
} = {}) {
  context.clearRect(0, 0, width, height);
  const corners = points?.length === 4 ? points : region ? regionToCorners(region) : null;
  if (!corners) return;
  const screenPoint = (horizontal, vertical) => {
    const point = pointInQuad(corners, horizontal, vertical);
    return { x: point.x * width, y: point.y * height };
  };
  const topLeft = screenPoint(0, 0);
  context.save();
  context.lineWidth = Math.max(2, width / 420);
  context.strokeStyle = calibrated ? 'rgba(115, 231, 189, 0.95)' : 'rgba(158, 143, 255, 0.95)';
  context.setLineDash(calibrated ? [] : [9, 7]);
  context.beginPath();
  corners.forEach((point, index) => {
    const x = point.x * width;
    const y = point.y * height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.closePath();
  context.stroke();
  context.setLineDash([]);

  if (calibrated && geometry) {
    context.font = `700 ${Math.max(10, width / 70)}px system-ui`;
    context.textAlign = 'center';
    context.textBaseline = 'bottom';
    for (const key of geometry.whites) {
      const top = screenPoint(key.xStart, 0);
      const bottom = screenPoint(key.xStart, 1);
      context.strokeStyle = 'rgba(220, 222, 255, 0.25)';
      context.beginPath();
      context.moveTo(top.x, top.y);
      context.lineTo(bottom.x, bottom.y);
      context.stroke();
    }
    for (const key of pressed) {
      const zone = [
        screenPoint(key.xStart, key.yStart),
        screenPoint(key.xEnd, key.yStart),
        screenPoint(key.xEnd, key.yEnd),
        screenPoint(key.xStart, key.yEnd),
      ];
      context.fillStyle = key.type === 'black' ? 'rgba(232, 75, 178, 0.48)' : 'rgba(114, 103, 255, 0.42)';
      context.beginPath();
      zone.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
      context.closePath();
      context.fill();
      context.fillStyle = '#ffffff';
      const labelPoint = screenPoint(key.xCenter, key.yEnd);
      context.fillText(key.note, labelPoint.x, labelPoint.y - 3);
    }
  }

  manualPoints.forEach((point, index) => {
    const x = point.x * width;
    const y = point.y * height;
    context.beginPath();
    context.arc(x, y, Math.max(9, width / 75), 0, Math.PI * 2);
    context.fillStyle = '#f0cc4f';
    context.fill();
    context.fillStyle = '#17172f';
    context.font = `900 ${Math.max(10, width / 72)}px system-ui`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(String(index + 1), x, y);
  });

  const label = calibrated
    ? 'Calibrated keyboard'
    : `${detected ? 'Keyboard pattern' : 'Searching'} ${Math.round(confidence * 100)}%`;
  context.font = `800 ${Math.max(11, width / 62)}px system-ui`;
  context.textAlign = 'left';
  context.textBaseline = 'bottom';
  const labelWidth = context.measureText(label).width + 18;
  context.fillStyle = 'rgba(16, 17, 39, 0.78)';
  context.fillRect(topLeft.x, Math.max(0, topLeft.y - 28), labelWidth, 25);
  context.fillStyle = '#f4f2ff';
  context.fillText(label, topLeft.x + 9, Math.max(20, topLeft.y - 8));
  context.restore();
}
