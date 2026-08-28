#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import ToneMidi from '@tonejs/midi';

const { Midi } = ToneMidi;

const PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const EPSILON = 1e-9;

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function pitchClass(midi) {
  return ((Math.round(midi) % 12) + 12) % 12;
}

function noteName(midi) {
  const rounded = Math.round(midi);
  return `${PITCH_CLASSES[pitchClass(rounded)]}${Math.floor(rounded / 12) - 1}`;
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function quantile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = clamp(fraction, 0, 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const mix = position - lower;
  return sorted[lower] * (1 - mix) + sorted[upper] * mix;
}

function seededRandom(seed = 0x504f4c59) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function isPercussiveInstrument(instrument) {
  return /drum|percussion|cymbal|kick|snare|hi[-_ ]?hat/i.test(String(instrument || ''));
}

export function normalizeNotes(notes, { dropPercussion = false } = {}) {
  const normalized = (Array.isArray(notes) ? notes : [])
    .map((note, sourceIndex) => {
      const midi = Math.round(Number(note?.midi ?? note?.pitch));
      const time = Number(note?.time ?? note?.startTime ?? note?.start);
      const duration = Math.max(0.01, Number(note?.duration ?? 0.1));
      const velocity = clamp(Number(note?.velocity ?? 0.75), 0, 1);
      const instrument = String(note?.instrument ?? note?.trackName ?? 'unknown');
      if (!Number.isFinite(midi) || midi < 0 || midi > 127 || !Number.isFinite(time) || time < 0) {
        return null;
      }
      return {
        sourceIndex,
        midi,
        pitchClass: pitchClass(midi),
        note: noteName(midi),
        time,
        duration: Number.isFinite(duration) ? duration : 0.1,
        velocity: Number.isFinite(velocity) ? velocity : 0.75,
        instrument,
        percussive: isPercussiveInstrument(instrument),
      };
    })
    .filter(Boolean)
    .filter((note) => !dropPercussion || !note.percussive)
    .sort((a, b) => a.time - b.time || a.midi - b.midi);

  // MuScriptor may emit the same pitch several frames apart. Those duplicates
  // are weak evidence and can create false anchors, so retain the strongest one.
  const deduplicated = [];
  for (const note of normalized) {
    const previous = deduplicated.at(-1);
    if (previous && previous.midi === note.midi && Math.abs(previous.time - note.time) <= 0.04) {
      if (note.duration * note.velocity > previous.duration * previous.velocity) {
        deduplicated[deduplicated.length - 1] = note;
      }
      continue;
    }
    deduplicated.push(note);
  }
  return deduplicated;
}

function sampleEvenly(items, limit) {
  if (items.length <= limit) return items;
  return Array.from({ length: limit }, (_, index) => {
    const position = Math.round((index / Math.max(1, limit - 1)) * (items.length - 1));
    return items[position];
  });
}

function indexByPitchClass(notes) {
  const index = Array.from({ length: 12 }, () => []);
  for (const note of notes) index[note.pitchClass].push(note);
  return index;
}

function closestByTime(notes, targetTime) {
  if (!notes.length) return null;
  let low = 0;
  let high = notes.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (notes[middle].time < targetTime) low = middle + 1;
    else high = middle;
  }
  const candidates = [notes[low - 1], notes[low], notes[low + 1]].filter(Boolean);
  return candidates.reduce((best, note) => (
    !best || Math.abs(note.time - targetTime) < Math.abs(best.time - targetTime) ? note : best
  ), null);
}

function scoreLine(referenceSample, observedByPitchClass, scale, offset, tolerance) {
  let score = 0;
  let supported = 0;
  const timeResiduals = [];
  for (const reference of referenceSample) {
    const expectedTime = reference.time * scale + offset;
    const observed = closestByTime(observedByPitchClass[reference.pitchClass], expectedTime);
    if (!observed) continue;
    const residual = Math.abs(observed.time - expectedTime);
    if (residual > tolerance) continue;
    const octaveDistance = Math.abs(observed.midi - reference.midi) / 12;
    const pitchWeight = observed.midi === reference.midi
      ? 1.25
      : Math.max(0.5, 0.92 - octaveDistance * 0.08);
    const timeWeight = 1 - residual / (tolerance + EPSILON);
    score += pitchWeight * (0.35 + 0.65 * timeWeight);
    supported += 1;
    timeResiduals.push(residual);
  }
  const coverage = supported / Math.max(1, referenceSample.length);
  return {
    score: score * (0.3 + 0.7 * coverage),
    supported,
    coverage,
    medianResidual: median(timeResiduals) ?? Infinity,
  };
}

function estimateCoarseLine(reference, observed, options) {
  const referenceSample = sampleEvenly(reference, options.ransacReferenceLimit);
  const observedSample = sampleEvenly(observed, options.ransacObservedLimit);
  const observedByPitchClass = indexByPitchClass(observed);
  const sampledObservedByPitchClass = indexByPitchClass(observedSample);
  const random = seededRandom(options.seed);
  const referenceSpan = Math.max(EPSILON, reference.at(-1).time - reference[0].time);
  const minimumSeparation = Math.max(3, referenceSpan * 0.12);
  let best = null;

  for (let iteration = 0; iteration < options.ransacIterations; iteration += 1) {
    const firstReference = referenceSample[Math.floor(random() * referenceSample.length)];
    const secondReference = referenceSample[Math.floor(random() * referenceSample.length)];
    if (!firstReference || !secondReference) continue;
    if (Math.abs(secondReference.time - firstReference.time) < minimumSeparation) continue;

    const firstPool = sampledObservedByPitchClass[firstReference.pitchClass];
    const secondPool = sampledObservedByPitchClass[secondReference.pitchClass];
    if (!firstPool.length || !secondPool.length) continue;
    const firstObserved = firstPool[Math.floor(random() * firstPool.length)];
    const secondObserved = secondPool[Math.floor(random() * secondPool.length)];
    const referenceDelta = secondReference.time - firstReference.time;
    const observedDelta = secondObserved.time - firstObserved.time;
    const scale = observedDelta / referenceDelta;
    if (!Number.isFinite(scale) || scale < options.minimumScale || scale > options.maximumScale) continue;
    const offset = firstObserved.time - firstReference.time * scale;
    const result = scoreLine(
      referenceSample,
      observedByPitchClass,
      scale,
      offset,
      options.coarseTolerance,
    );
    const candidate = { scale, offset, ...result };
    if (
      !best
      || candidate.score > best.score
      || (candidate.score === best.score && candidate.medianResidual < best.medianResidual)
    ) {
      best = candidate;
    }
  }

  if (!best || best.supported < options.minimumCoarseSupport) {
    throw new Error('Could not find a reliable musical sequence shared by MuScriptor and the MIDI.');
  }
  return best;
}

function candidateCost(reference, observed, expectedTime, windowSeconds) {
  const timeCost = Math.abs(observed.time - expectedTime) / windowSeconds;
  const octaveDistance = Math.abs(observed.midi - reference.midi) / 12;
  const pitchCost = observed.midi === reference.midi ? 0 : 0.18 + octaveDistance * 0.035;
  const durationRatio = Math.max(reference.duration, observed.duration)
    / Math.max(0.03, Math.min(reference.duration, observed.duration));
  const durationCost = Math.min(0.14, Math.abs(Math.log(durationRatio)) * 0.025);
  return timeCost + pitchCost + durationCost;
}

function collectMatches(reference, observed, line, options) {
  const byPitchClass = indexByPitchClass(observed);
  const usedObserved = new Set();
  const matches = [];
  let lastObservedTime = -Infinity;

  for (let referenceIndex = 0; referenceIndex < reference.length; referenceIndex += 1) {
    const target = reference[referenceIndex];
    const expectedTime = Array.isArray(line.anchors) && line.anchors.length >= 2
      ? mapReferenceTime(target.time, line.anchors)
      : target.time * line.scale + line.offset;
    const pool = byPitchClass[target.pitchClass];
    let best = null;
    for (const candidate of pool) {
      if (candidate.time < expectedTime - options.matchWindowSeconds) continue;
      if (candidate.time > expectedTime + options.matchWindowSeconds) break;
      if (candidate.time < lastObservedTime - options.chordToleranceSeconds) continue;
      if (usedObserved.has(candidate.sourceIndex)) continue;
      const cost = candidateCost(target, candidate, expectedTime, options.matchWindowSeconds);
      if (!best || cost < best.cost) best = { candidate, cost };
    }
    if (!best || best.cost > options.maximumMatchCost) continue;
    usedObserved.add(best.candidate.sourceIndex);
    lastObservedTime = Math.max(lastObservedTime, best.candidate.time);
    matches.push({
      referenceIndex,
      observedIndex: best.candidate.sourceIndex,
      reference: target,
      observed: best.candidate,
      expectedTime,
      coarseResidual: best.candidate.time - expectedTime,
      exactPitch: best.candidate.midi === target.midi,
      octaveDifference: Math.round((best.candidate.midi - target.midi) / 12),
      cost: best.cost,
    });
  }
  return matches;
}

function cosineSimilarity(left, right) {
  let dot = 0;
  let leftEnergy = 0;
  let rightEnergy = 0;
  for (let index = 0; index < 12; index += 1) {
    dot += left[index] * right[index];
    leftEnergy += left[index] ** 2;
    rightEnergy += right[index] ** 2;
  }
  if (leftEnergy <= EPSILON || rightEnergy <= EPSILON) return null;
  return dot / Math.sqrt(leftEnergy * rightEnergy);
}

function buildChromaFrames(notes, frameSeconds) {
  const maximumTime = Math.max(...notes.map((note) => note.time + note.duration), 0);
  const frames = Array.from(
    { length: Math.ceil(maximumTime / frameSeconds) + 2 },
    () => new Float64Array(12),
  );
  for (const note of notes) {
    const start = Math.max(0, Math.floor(note.time / frameSeconds));
    const limitedEnd = Math.min(note.time + note.duration, note.time + 2.5);
    const end = Math.min(frames.length - 1, Math.floor(limitedEnd / frameSeconds));
    const strength = 0.35 + note.velocity * 0.65;
    for (let frameIndex = start; frameIndex <= end; frameIndex += 1) {
      const onsetWeight = frameIndex === start ? 1 : 0.28;
      frames[frameIndex][note.pitchClass] += strength * onsetWeight;
    }
  }
  return frames.map((frame, index) => {
    const smoothed = new Float64Array(12);
    for (let pitch = 0; pitch < 12; pitch += 1) {
      smoothed[pitch] = frame[pitch] * 0.6
        + (frames[index - 1]?.[pitch] || 0) * 0.2
        + (frames[index + 1]?.[pitch] || 0) * 0.2;
    }
    return smoothed;
  });
}

function structuralWindowSimilarity(
  referenceFrames,
  observedFrames,
  referenceCentre,
  observedCentre,
  scale,
  options,
) {
  const frameSeconds = Number(options.chromaFrameSeconds) || 0.25;
  const radius = Number(options.chromaWindowRadiusSeconds) || 3.5;
  const steps = Math.max(2, Math.round(radius / frameSeconds));
  const similarities = [];
  for (let step = -steps; step <= steps; step += 1) {
    const referenceTime = referenceCentre + step * frameSeconds;
    const observedTime = observedCentre + step * frameSeconds * scale;
    if (referenceTime < 0 || observedTime < 0) continue;
    const referenceFrame = referenceFrames[Math.round(referenceTime / frameSeconds)];
    const observedFrame = observedFrames[Math.round(observedTime / frameSeconds)];
    if (!referenceFrame || !observedFrame) continue;
    const similarity = cosineSimilarity(referenceFrame, observedFrame);
    if (similarity != null) similarities.push(similarity);
  }
  if (!similarities.length) return { similarity: 0, support: 0 };
  return {
    similarity: similarities.reduce((sum, value) => sum + value, 0) / similarities.length,
    support: similarities.length,
  };
}

function estimateStructuralOffsetPath(reference, observed, line, options) {
  const frameSeconds = Number(options.chromaFrameSeconds) || 0.25;
  const stepSeconds = Math.max(2, Number(options.anchorBinSeconds) || 4);
  const searchSeconds = Math.max(2, Number(options.chromaSearchSeconds) || 8);
  const searchStep = Math.max(frameSeconds, Number(options.chromaSearchStepSeconds) || 0.25);
  const referenceFrames = buildChromaFrames(reference, frameSeconds);
  const observedFrames = buildChromaFrames(observed, frameSeconds);
  const maximumReferenceTime = reference.at(-1)?.time ?? 0;
  const bins = [];

  for (let centre = stepSeconds / 2; centre <= maximumReferenceTime; centre += stepSeconds) {
    const candidates = [];
    for (let delta = -searchSeconds; delta <= searchSeconds + EPSILON; delta += searchStep) {
      const observedCentre = centre * line.scale + line.offset + delta;
      const result = structuralWindowSimilarity(
        referenceFrames,
        observedFrames,
        centre,
        observedCentre,
        line.scale,
        options,
      );
      candidates.push({
        delta,
        emission: result.similarity * 5 + Math.min(1, result.support / 12),
        ...result,
      });
    }
    bins.push({ centre, candidates });
  }
  if (bins.length < 2) return [];

  let previous = bins[0].candidates.map((candidate) => ({
    score: candidate.emission - Math.abs(candidate.delta) * 0.04,
    previousIndex: -1,
  }));
  const backPointers = [previous];
  for (let binIndex = 1; binIndex < bins.length; binIndex += 1) {
    const current = bins[binIndex].candidates.map((candidate) => {
      let best = { score: -Infinity, previousIndex: -1 };
      bins[binIndex - 1].candidates.forEach((priorCandidate, priorIndex) => {
        const change = Math.abs(candidate.delta - priorCandidate.delta);
        const transitionPenalty = change * (Number(options.chromaSmoothnessPenalty) || 0.42)
          + Math.max(0, change - 1.5) * 0.55;
        const score = previous[priorIndex].score + candidate.emission - transitionPenalty;
        if (score > best.score) best = { score, previousIndex: priorIndex };
      });
      return best;
    });
    previous = current;
    backPointers.push(current);
  }

  let candidateIndex = previous.reduce((bestIndex, state, index) => (
    state.score > previous[bestIndex].score ? index : bestIndex
  ), 0);
  const path = [];
  for (let binIndex = bins.length - 1; binIndex >= 0; binIndex -= 1) {
    const bin = bins[binIndex];
    const candidate = bin.candidates[candidateIndex];
    path.push({ bin, candidate });
    candidateIndex = backPointers[binIndex][candidateIndex].previousIndex;
    if (candidateIndex < 0 && binIndex > 0) candidateIndex = 0;
  }
  path.reverse();

  const anchors = path
    .filter(({ candidate }) => candidate.support >= 6 && candidate.similarity >= 0.24)
    .map(({ bin, candidate }) => ({
      referenceTime: bin.centre,
      observedTime: bin.centre * line.scale + line.offset + candidate.delta,
      support: candidate.support,
      exactPitchShare: null,
      structuralSimilarity: Number(candidate.similarity.toFixed(4)),
      localOffsetSeconds: Number(candidate.delta.toFixed(4)),
      kind: 'automatic-chroma',
    }));
  const monotonic = [];
  for (const anchor of anchors) {
    if (!monotonic.length || anchor.observedTime > monotonic.at(-1).observedTime + 0.02) {
      monotonic.push(anchor);
    }
  }
  return monotonic;
}

function estimateLocalOffsetPath(reference, observed, line, options) {
  const stepSeconds = Math.max(2, Number(options.anchorBinSeconds) || 4);
  const radiusSeconds = Math.max(stepSeconds * 0.8, Number(options.localSearchRadiusSeconds) || 3.2);
  const searchSeconds = Math.max(1, Number(options.localOffsetSearchSeconds) || 6);
  const searchStep = Math.max(0.05, Number(options.localOffsetStepSeconds) || 0.15);
  const tolerance = Math.max(0.08, Number(options.localSearchToleranceSeconds) || 0.28);
  const observedByPitchClass = indexByPitchClass(observed);
  const maximumReferenceTime = reference.at(-1)?.time ?? 0;
  const bins = [];

  for (let centre = stepSeconds / 2; centre <= maximumReferenceTime; centre += stepSeconds) {
    const localReference = sampleEvenly(reference.filter((note) => (
      note.time >= centre - radiusSeconds && note.time <= centre + radiusSeconds
    )), 36);
    if (localReference.length < 4) continue;
    const candidates = [];
    for (let delta = -searchSeconds; delta <= searchSeconds + EPSILON; delta += searchStep) {
      const result = scoreLine(
        localReference,
        observedByPitchClass,
        line.scale,
        line.offset + delta,
        tolerance,
      );
      candidates.push({
        delta,
        emission: result.score + result.coverage * 2.5 - result.medianResidual * 1.5,
        ...result,
      });
    }
    bins.push({ centre, localReferenceCount: localReference.length, candidates });
  }
  if (bins.length < 2) return [];

  let previous = bins[0].candidates.map((candidate) => ({
    score: candidate.emission - Math.abs(candidate.delta) * 0.06,
    previousIndex: -1,
  }));
  const backPointers = [previous];
  for (let binIndex = 1; binIndex < bins.length; binIndex += 1) {
    const current = bins[binIndex].candidates.map((candidate) => {
      let best = { score: -Infinity, previousIndex: -1 };
      bins[binIndex - 1].candidates.forEach((priorCandidate, priorIndex) => {
        const change = Math.abs(candidate.delta - priorCandidate.delta);
        const transitionPenalty = change * (Number(options.localOffsetSmoothnessPenalty) || 0.72)
          + Math.max(0, change - 1.5) * 0.35;
        const score = previous[priorIndex].score + candidate.emission - transitionPenalty;
        if (score > best.score) best = { score, previousIndex: priorIndex };
      });
      return best;
    });
    previous = current;
    backPointers.push(current);
  }

  let candidateIndex = previous.reduce((bestIndex, state, index) => (
    state.score > previous[bestIndex].score ? index : bestIndex
  ), 0);
  const path = [];
  for (let binIndex = bins.length - 1; binIndex >= 0; binIndex -= 1) {
    const bin = bins[binIndex];
    const candidate = bin.candidates[candidateIndex];
    path.push({ bin, candidate });
    candidateIndex = backPointers[binIndex][candidateIndex].previousIndex;
    if (candidateIndex < 0 && binIndex > 0) candidateIndex = 0;
  }
  path.reverse();

  const anchors = path
    .filter(({ bin, candidate }) => (
      candidate.supported >= Math.max(3, Math.ceil(bin.localReferenceCount * 0.18))
      && candidate.coverage >= 0.18
    ))
    .map(({ bin, candidate }) => ({
      referenceTime: bin.centre,
      observedTime: bin.centre * line.scale + line.offset + candidate.delta,
      support: candidate.supported,
      exactPitchShare: null,
      medianLineResidualMs: Number((candidate.medianResidual * 1000).toFixed(2)),
      localOffsetSeconds: Number(candidate.delta.toFixed(4)),
      kind: 'automatic-local-search',
    }));
  const monotonic = [];
  for (const anchor of anchors) {
    if (!monotonic.length || anchor.observedTime > monotonic.at(-1).observedTime + 0.02) {
      monotonic.push(anchor);
    }
  }
  return monotonic;
}

function mergeAutomaticAnchors(matchAnchors, localAnchors, options) {
  if (localAnchors.length < 2) return matchAnchors;
  const radius = Math.max(0.5, (Number(options.anchorBinSeconds) || 4) * 0.45);
  const combined = [
    ...matchAnchors.filter((matchAnchor) => !localAnchors.some((localAnchor) => (
      Math.abs(localAnchor.referenceTime - matchAnchor.referenceTime) <= radius
    ))),
    ...localAnchors,
  ].sort((a, b) => a.referenceTime - b.referenceTime || b.support - a.support);
  const monotonic = [];
  for (const anchor of combined) {
    const previous = monotonic.at(-1);
    if (!previous) monotonic.push(anchor);
    else if (
      anchor.referenceTime > previous.referenceTime + 0.02
      && anchor.observedTime > previous.observedTime + 0.02
    ) monotonic.push(anchor);
  }
  return monotonic.length >= 2 ? monotonic : matchAnchors;
}

function robustLinearFit(matches, fallback) {
  if (matches.length < 3) return fallback;
  let retained = [...matches];
  for (let pass = 0; pass < 3; pass += 1) {
    const meanX = retained.reduce((sum, match) => sum + match.reference.time, 0) / retained.length;
    const meanY = retained.reduce((sum, match) => sum + match.observed.time, 0) / retained.length;
    const denominator = retained.reduce(
      (sum, match) => sum + (match.reference.time - meanX) ** 2,
      0,
    );
    const scale = denominator > EPSILON
      ? retained.reduce(
        (sum, match) => sum + (match.reference.time - meanX) * (match.observed.time - meanY),
        0,
      ) / denominator
      : fallback.scale;
    const offset = meanY - scale * meanX;
    const residuals = retained.map((match) => match.observed.time - (scale * match.reference.time + offset));
    const centre = median(residuals) ?? 0;
    const mad = median(residuals.map((residual) => Math.abs(residual - centre))) ?? 0;
    const threshold = Math.max(0.12, mad * 3.5);
    const filtered = retained.filter((match) => (
      Math.abs(match.observed.time - (scale * match.reference.time + offset) - centre) <= threshold
    ));
    const fitted = { scale, offset };
    if (filtered.length === retained.length || filtered.length < 3) return fitted;
    retained = filtered;
  }
  return fallback;
}

function buildWarpAnchors(matches, line, options) {
  const candidates = matches
    .filter((match) => Math.abs(match.coarseResidual) <= options.localAnchorTolerance)
    .sort((a, b) => a.reference.time - b.reference.time || a.observed.time - b.observed.time);
  if (candidates.length < 3) {
    return [
      { referenceTime: 0, observedTime: line.offset, support: 0, kind: 'linear-fallback' },
      {
        referenceTime: Math.max(1, matches.at(-1)?.reference.time ?? 1),
        observedTime: Math.max(1, matches.at(-1)?.reference.time ?? 1) * line.scale + line.offset,
        support: 0,
        kind: 'linear-fallback',
      },
    ];
  }

  const bins = new Map();
  for (const match of candidates) {
    const bin = Math.floor(match.reference.time / options.anchorBinSeconds);
    const entries = bins.get(bin) || [];
    entries.push(match);
    bins.set(bin, entries);
  }

  const anchors = [...bins.values()].map((entries) => ({
    referenceTime: median(entries.map((match) => match.reference.time)),
    observedTime: median(entries.map((match) => match.observed.time)),
    support: entries.length,
    exactPitchShare: entries.filter((match) => match.exactPitch).length / entries.length,
    medianLineResidualMs: Number((median(entries.map((match) => (
      Math.abs(match.observed.time - (line.scale * match.reference.time + line.offset))
    ))) * 1000).toFixed(2)),
    kind: 'automatic',
  })).sort((a, b) => a.referenceTime - b.referenceTime);

  // The mapping must always move forward. A backward anchor is normally a
  // repeated-chorus mismatch or a noise note and is unsafe for training labels.
  const monotonic = [];
  for (const anchor of anchors) {
    const previous = monotonic.at(-1);
    if (!previous || anchor.observedTime > previous.observedTime + 0.01) monotonic.push(anchor);
  }
  return monotonic.length >= 2 ? monotonic : buildWarpAnchors([], line, options);
}

function normalizeManualAnchors(input, maximumReferenceTime, sourceDurationSeconds) {
  const anchors = (Array.isArray(input) ? input : [])
    .map((anchor, index) => ({
      referenceTime: Number(anchor?.referenceTime ?? anchor?.reference ?? anchor?.midiTime),
      observedTime: Number(anchor?.observedTime ?? anchor?.observed ?? anchor?.sourceTime),
      support: Number.MAX_SAFE_INTEGER,
      exactPitchShare: 1,
      medianLineResidualMs: 0,
      kind: 'manual',
      manualIndex: index,
    }))
    .filter((anchor) => Number.isFinite(anchor.referenceTime) && Number.isFinite(anchor.observedTime))
    .filter((anchor) => anchor.referenceTime >= 0 && anchor.referenceTime <= maximumReferenceTime + 1)
    .filter((anchor) => anchor.observedTime >= 0 && (
      !Number.isFinite(sourceDurationSeconds) || anchor.observedTime <= sourceDurationSeconds + 0.05
    ))
    .sort((a, b) => a.referenceTime - b.referenceTime || a.observedTime - b.observedTime);

  for (let index = 1; index < anchors.length; index += 1) {
    if (anchors[index].referenceTime <= anchors[index - 1].referenceTime + EPSILON) {
      throw new Error('Manual MIDI/reference anchor times must increase without duplicates.');
    }
    if (anchors[index].observedTime <= anchors[index - 1].observedTime + EPSILON) {
      throw new Error('Manual source/video anchor times must increase. A timeline cannot move backwards.');
    }
  }
  return anchors;
}

function mergeWarpAnchors(automaticAnchors, manualAnchors, line, maximumReferenceTime, options) {
  if (!manualAnchors.length) return automaticAnchors;
  const radius = Math.max(0.1, Number(options.manualAnchorRadiusSeconds) || 2.5);
  const candidates = [
    ...automaticAnchors.filter((automatic) => !manualAnchors.some((manual) => (
      Math.abs(manual.referenceTime - automatic.referenceTime) <= radius
    ))),
    ...manualAnchors,
  ].sort((a, b) => a.referenceTime - b.referenceTime || (
    a.kind === 'manual' ? -1 : 1
  ));

  const monotonic = [];
  for (const anchor of candidates) {
    const previous = monotonic.at(-1);
    if (!previous) {
      monotonic.push(anchor);
      continue;
    }
    if (
      anchor.referenceTime > previous.referenceTime + EPSILON
      && anchor.observedTime > previous.observedTime + 0.005
    ) {
      monotonic.push(anchor);
      continue;
    }
    if (anchor.kind !== 'manual') continue;
    while (
      monotonic.length
      && monotonic.at(-1).kind !== 'manual'
      && (
        anchor.referenceTime <= monotonic.at(-1).referenceTime + EPSILON
        || anchor.observedTime <= monotonic.at(-1).observedTime + 0.005
      )
    ) monotonic.pop();
    const last = monotonic.at(-1);
    if (
      last
      && (
        anchor.referenceTime <= last.referenceTime + EPSILON
        || anchor.observedTime <= last.observedTime + 0.005
      )
    ) {
      throw new Error('Manual anchors conflict with each other after automatic outliers are removed.');
    }
    monotonic.push(anchor);
  }

  if (monotonic.length >= 2) return monotonic;
  return [
    { referenceTime: 0, observedTime: line.offset, support: 0, kind: 'linear-fallback' },
    {
      referenceTime: Math.max(1, maximumReferenceTime),
      observedTime: Math.max(1, maximumReferenceTime) * line.scale + line.offset,
      support: 0,
      kind: 'linear-fallback',
    },
  ];
}

export function mapReferenceTime(referenceTime, anchors) {
  if (!anchors.length) return referenceTime;
  if (anchors.length === 1) return anchors[0].observedTime + referenceTime - anchors[0].referenceTime;
  let left = anchors[0];
  let right = anchors[1];
  if (referenceTime <= anchors[0].referenceTime) {
    [left, right] = [anchors[0], anchors[1]];
  } else if (referenceTime >= anchors.at(-1).referenceTime) {
    [left, right] = [anchors.at(-2), anchors.at(-1)];
  } else {
    for (let index = 1; index < anchors.length; index += 1) {
      if (referenceTime <= anchors[index].referenceTime) {
        [left, right] = [anchors[index - 1], anchors[index]];
        break;
      }
    }
  }
  const scale = (right.observedTime - left.observedTime)
    / Math.max(EPSILON, right.referenceTime - left.referenceTime);
  return left.observedTime + (referenceTime - left.referenceTime) * scale;
}

function referenceRangeExcluded(start, end, ranges) {
  return (Array.isArray(ranges) ? ranges : []).some((range) => {
    const rangeStart = Number(range?.start ?? range?.referenceStart);
    const rangeEnd = Number(range?.end ?? range?.referenceEnd);
    return Number.isFinite(rangeStart) && Number.isFinite(rangeEnd)
      && Math.max(start, Math.min(rangeStart, rangeEnd)) < Math.min(end, Math.max(rangeStart, rangeEnd));
  });
}

function decisionForWindow(windowId, decisions) {
  if (Array.isArray(decisions)) {
    return String(decisions.find((item) => item?.windowId === windowId)?.decision || 'auto');
  }
  if (decisions && typeof decisions === 'object') return String(decisions[windowId] || 'auto');
  return 'auto';
}

function slopeBetween(start, end, anchors) {
  return (mapReferenceTime(end, anchors) - mapReferenceTime(start, anchors))
    / Math.max(EPSILON, end - start);
}

function buildQualityWindows(reference, matches, anchors, coarseLine, options) {
  const windowSeconds = Math.max(2, Number(options.qualityWindowSeconds) || 5);
  const maximumReferenceTime = Math.max(...reference.map((note) => note.time + note.duration), 0);
  const sourceDuration = options.sourceDurationSeconds == null || options.sourceDurationSeconds === ''
    ? Number.NaN
    : Number(options.sourceDurationSeconds);
  const matchesByReference = new Map(matches.map((match) => [match.referenceIndex, match]));
  const windows = [];
  for (let start = 0, index = 0; start < maximumReferenceTime + EPSILON; start += windowSeconds, index += 1) {
    const end = Math.min(maximumReferenceTime, start + windowSeconds);
    if (end <= start + EPSILON) break;
    const entries = reference
      .map((note, referenceIndex) => ({ note, referenceIndex }))
      .filter(({ note }) => note.time >= start && note.time < end);
    const windowMatches = entries.map(({ referenceIndex }) => matchesByReference.get(referenceIndex)).filter(Boolean);
    const residuals = windowMatches.map((match) => Math.abs(
      match.observed.time - mapReferenceTime(match.reference.time, anchors)
    ));
    const exact = windowMatches.filter((match) => match.exactPitch).length;
    const localScale = slopeBetween(start, end, anchors);
    const matchPercent = entries.length ? windowMatches.length / entries.length : 0;
    const exactPercent = windowMatches.length ? exact / windowMatches.length : 0;
    const medianResidual = median(residuals) ?? Infinity;
    const p95Residual = quantile(residuals, 0.95) ?? Infinity;
    const structuralSimilarity = median(anchors
      .filter((anchor) => (
        Number.isFinite(anchor.structuralSimilarity)
        && anchor.referenceTime >= start - 2.5
        && anchor.referenceTime <= end + 2.5
      ))
      .map((anchor) => anchor.structuralSimilarity));
    const flags = [];
    if (entries.length === 0) flags.push('reference-silence');
    if (entries.length > 0 && matchPercent < 0.55) flags.push('weak-note-support');
    if (windowMatches.length > 0 && exactPercent < 0.35) flags.push('octave-or-pitch-disagreement');
    if (medianResidual > 0.25) flags.push('timing-residual');
    if (structuralSimilarity != null && structuralSimilarity < 0.42) {
      flags.push('weak-harmonic-structure');
    }
    if (localScale < 0.72 || localScale > 1.38) flags.push('strong-local-tempo-change-or-pause');
    else if (Math.abs(localScale - coarseLine.scale) > 0.12) flags.push('local-tempo-drift');
    const structurallyTrusted = structuralSimilarity != null
      && structuralSimilarity >= 0.62
      && matchPercent >= 0.22
      && localScale >= 0.72
      && localScale <= 1.38;
    const structurallyReviewable = structuralSimilarity != null
      && structuralSimilarity >= 0.42
      && matchPercent >= 0.15
      && localScale >= 0.5
      && localScale <= 1.8;
    const automaticStatus = entries.length === 0
      ? 'neutral'
      : (matchPercent >= 0.72 && medianResidual <= 0.18 && localScale >= 0.72 && localScale <= 1.38)
        || structurallyTrusted
        ? 'trusted'
        : (matchPercent >= 0.45 && medianResidual <= 0.45 && localScale >= 0.5 && localScale <= 1.8)
          || structurallyReviewable
          ? 'review'
          : 'unsafe';
    const windowId = `w${String(index + 1).padStart(4, '0')}`;
    const decision = referenceRangeExcluded(start, end, options.excludedRanges)
      ? 'reject'
      : decisionForWindow(windowId, options.reviewDecisions);
    const status = decision === 'accept'
      ? 'accepted-manually'
      : decision === 'reject'
        ? 'rejected'
        : automaticStatus;
    const mappedStart = Math.max(0, mapReferenceTime(start, anchors));
    const mappedEndRaw = Math.max(mappedStart, mapReferenceTime(end, anchors));
    const mappedEnd = Number.isFinite(sourceDuration)
      ? Math.min(sourceDuration, mappedEndRaw)
      : mappedEndRaw;
    windows.push({
      id: windowId,
      referenceStart: Number(start.toFixed(4)),
      referenceEnd: Number(end.toFixed(4)),
      sourceStart: Number(mappedStart.toFixed(4)),
      sourceEnd: Number(mappedEnd.toFixed(4)),
      referenceNotes: entries.length,
      matchedNotes: windowMatches.length,
      matchedPercent: Number((matchPercent * 100).toFixed(1)),
      exactPitchPercent: Number((exactPercent * 100).toFixed(1)),
      medianResidualMs: Number.isFinite(medianResidual) ? Number((medianResidual * 1000).toFixed(1)) : null,
      p95ResidualMs: Number.isFinite(p95Residual) ? Number((p95Residual * 1000).toFixed(1)) : null,
      structuralSimilarity: structuralSimilarity == null
        ? null
        : Number(structuralSimilarity.toFixed(4)),
      localScale: Number(localScale.toFixed(5)),
      localTempoDifferencePercent: Number(((localScale - 1) * 100).toFixed(2)),
      automaticStatus,
      decision,
      status,
      trainingEligible: status === 'trusted' || status === 'accepted-manually',
      flags,
    });
  }
  return windows;
}

function buildTempoSegments(anchors, matches, coarseLine) {
  return anchors.slice(1).map((right, index) => {
    const left = anchors[index];
    const referenceSeconds = right.referenceTime - left.referenceTime;
    const sourceSeconds = right.observedTime - left.observedTime;
    const localScale = sourceSeconds / Math.max(EPSILON, referenceSeconds);
    const segmentMatches = matches.filter((match) => (
      match.reference.time >= left.referenceTime && match.reference.time <= right.referenceTime
    ));
    const residuals = segmentMatches.map((match) => Math.abs(
      match.observed.time - mapReferenceTime(match.reference.time, [left, right])
    ));
    const flags = [];
    if (localScale < 0.72 || localScale > 1.38) flags.push('pause-cut-or-strong-tempo-change');
    else if (Math.abs(localScale - coarseLine.scale) > 0.12) flags.push('tempo-drift');
    if (segmentMatches.length < 3) flags.push('low-support');
    return {
      id: `s${String(index + 1).padStart(4, '0')}`,
      referenceStart: Number(left.referenceTime.toFixed(4)),
      referenceEnd: Number(right.referenceTime.toFixed(4)),
      sourceStart: Number(left.observedTime.toFixed(4)),
      sourceEnd: Number(right.observedTime.toFixed(4)),
      referenceSeconds: Number(referenceSeconds.toFixed(4)),
      sourceSeconds: Number(sourceSeconds.toFixed(4)),
      localScale: Number(localScale.toFixed(5)),
      localTempoDifferencePercent: Number(((localScale - 1) * 100).toFixed(2)),
      extraOrMissingSecondsVsCoarse: Number((sourceSeconds - referenceSeconds * coarseLine.scale).toFixed(4)),
      matchedNotes: segmentMatches.length,
      exactPitchPercent: Number((segmentMatches.filter((match) => match.exactPitch).length
        / Math.max(1, segmentMatches.length) * 100).toFixed(1)),
      medianResidualMs: Number(((median(residuals) ?? 0) * 1000).toFixed(1)),
      leftKind: left.kind,
      rightKind: right.kind,
      flags,
    };
  });
}

function calculateMetrics(reference, observed, matches, anchors, coarseLine) {
  const residuals = matches.map((match) => (
    match.observed.time - mapReferenceTime(match.reference.time, anchors)
  ));
  const absoluteResiduals = residuals.map(Math.abs);
  const exactPitchMatches = matches.filter((match) => match.exactPitch).length;
  const octaveMatches = matches.length - exactPitchMatches;
  const localSlopes = anchors.slice(1).map((anchor, index) => (
    (anchor.observedTime - anchors[index].observedTime)
    / Math.max(EPSILON, anchor.referenceTime - anchors[index].referenceTime)
  ));
  const suspiciousSlopes = localSlopes.filter((slope) => slope < 0.7 || slope > 1.3).length;
  const matchedReferenceShare = matches.length / Math.max(1, reference.length);
  const p95Residual = quantile(absoluteResiduals, 0.95) ?? Infinity;
  const confidence = clamp(
    matchedReferenceShare * 0.48
      + (exactPitchMatches / Math.max(1, matches.length)) * 0.22
      + Math.max(0, 1 - (median(absoluteResiduals) ?? 3) / 0.35) * 0.18
      + Math.max(0, 1 - p95Residual / 1.2) * 0.12
      - suspiciousSlopes * 0.03,
    0,
    1,
  );
  return {
    referenceNotes: reference.length,
    observedNotes: observed.length,
    matchedNotes: matches.length,
    unmatchedReferenceNotes: reference.length - matches.length,
    unmatchedObservedNotes: observed.length - new Set(matches.map((match) => match.observedIndex)).size,
    matchedReferencePercent: Number((matchedReferenceShare * 100).toFixed(2)),
    exactPitchMatches,
    octaveEquivalentMatches: octaveMatches,
    exactPitchPercent: Number((exactPitchMatches / Math.max(1, matches.length) * 100).toFixed(2)),
    medianTimingResidualMs: Number(((median(absoluteResiduals) ?? 0) * 1000).toFixed(2)),
    p95TimingResidualMs: Number((p95Residual * 1000).toFixed(2)),
    coarseScale: Number(coarseLine.scale.toFixed(8)),
    coarseOffsetSeconds: Number(coarseLine.offset.toFixed(6)),
    estimatedAverageSpeedDifferencePercent: Number(((coarseLine.scale - 1) * 100).toFixed(3)),
    localAnchorCount: anchors.length,
    suspiciousLocalTempoSegments: suspiciousSlopes,
    confidence: Number(confidence.toFixed(4)),
    verdict: confidence >= 0.82 && p95Residual <= 0.12
      ? 'review-likely-pass'
      : confidence >= 0.6
        ? 'manual-review-required'
        : 'reject-or-add-manual-anchors',
  };
}

const DEFAULT_OPTIONS = {
  seed: 0x504f4c59,
  ransacIterations: 18000,
  ransacReferenceLimit: 100,
  ransacObservedLimit: 150,
  minimumScale: 0.65,
  maximumScale: 1.45,
  coarseTolerance: 1.6,
  minimumCoarseSupport: 8,
  matchWindowSeconds: 1.4,
  maximumMatchCost: 1.25,
  chordToleranceSeconds: 0.08,
  localAnchorTolerance: 1.35,
  anchorBinSeconds: 4,
  localSearchRadiusSeconds: 3.2,
  localOffsetSearchSeconds: 6,
  localOffsetStepSeconds: 0.15,
  localSearchToleranceSeconds: 0.28,
  localOffsetSmoothnessPenalty: 0.72,
  chromaFrameSeconds: 0.25,
  chromaWindowRadiusSeconds: 3.5,
  chromaSearchSeconds: 8,
  chromaSearchStepSeconds: 0.25,
  chromaSmoothnessPenalty: 0.42,
  manualAnchorRadiusSeconds: 2.5,
  qualityWindowSeconds: 5,
  sourceDurationSeconds: null,
  manualAnchors: [],
  excludedRanges: [],
  reviewDecisions: {},
};

export function alignNoteCoordinates(referenceInput, observedInput, overrides = {}) {
  const options = { ...DEFAULT_OPTIONS, ...overrides };
  const reference = normalizeNotes(referenceInput, { dropPercussion: true });
  const observed = normalizeNotes(observedInput, { dropPercussion: true });
  if (reference.length < 8 || observed.length < 8) {
    throw new Error('At least eight usable notes are required in both files.');
  }
  const coarse = estimateCoarseLine(reference, observed, options);
  let line = coarse;
  let matches = [];
  for (let pass = 0; pass < 3; pass += 1) {
    matches = collectMatches(reference, observed, line, options);
    line = { ...line, ...robustLinearFit(matches, line) };
  }
  matches = collectMatches(reference, observed, line, options);
  const maximumReferenceTime = Math.max(...reference.map((note) => note.time + note.duration), 1);
  const sourceDurationSeconds = options.sourceDurationSeconds == null || options.sourceDurationSeconds === ''
    ? Number.NaN
    : Number(options.sourceDurationSeconds);
  const manualAnchors = normalizeManualAnchors(
    options.manualAnchors,
    maximumReferenceTime,
    sourceDurationSeconds,
  );
  const localSearchAnchors = estimateLocalOffsetPath(reference, observed, line, options);
  const structuralAnchors = estimateStructuralOffsetPath(reference, observed, line, options);
  const searchAnchors = structuralAnchors.length >= 2
    ? structuralAnchors
    : localSearchAnchors;
  const initialAutomaticAnchors = mergeAutomaticAnchors(
    buildWarpAnchors(matches, line, options),
    searchAnchors,
    options,
  );
  const preliminaryAnchors = mergeWarpAnchors(
    initialAutomaticAnchors,
    manualAnchors,
    line,
    maximumReferenceTime,
    options,
  );
  matches = collectMatches(reference, observed, { ...line, anchors: preliminaryAnchors }, options);
  const automaticAnchors = mergeAutomaticAnchors(
    buildWarpAnchors(matches, line, options),
    searchAnchors,
    options,
  );
  const anchors = mergeWarpAnchors(
    automaticAnchors,
    manualAnchors,
    line,
    maximumReferenceTime,
    options,
  ).filter((anchor) => (
    !Number.isFinite(sourceDurationSeconds) || anchor.observedTime <= sourceDurationSeconds + 0.05
  ));
  if (anchors.length < 2) throw new Error('Not enough forward-moving anchors remain inside the source timeline.');
  matches = collectMatches(reference, observed, { ...line, anchors }, options);
  const qualityWindows = buildQualityWindows(reference, matches, anchors, line, options);
  const tempoSegments = buildTempoSegments(anchors, matches, line);
  const windowByReferenceTime = (time) => qualityWindows.find((window) => (
    time >= window.referenceStart && time < window.referenceEnd + EPSILON
  ));
  const alignedReference = reference.map((note) => {
    const mappedStartRaw = mapReferenceTime(note.time, anchors);
    const mappedEndRaw = mapReferenceTime(note.time + note.duration, anchors);
    const mappedStart = Math.max(0, mappedStartRaw);
    const mappedEnd = Number.isFinite(sourceDurationSeconds)
      ? Math.min(sourceDurationSeconds, Math.max(mappedStart, mappedEndRaw))
      : Math.max(mappedStart, mappedEndRaw);
    if (Number.isFinite(sourceDurationSeconds) && mappedStart >= sourceDurationSeconds - 0.005) return null;
    const quality = windowByReferenceTime(note.time);
    return {
      ...note,
      originalTime: note.time,
      time: Number(mappedStart.toFixed(6)),
      duration: Number(Math.max(0.01, mappedEnd - mappedStart).toFixed(6)),
      qualityWindowId: quality?.id || null,
      qualityStatus: quality?.status || 'unsafe',
      trainingEligible: Boolean(quality?.trainingEligible),
    };
  }).filter(Boolean);
  const baseMetrics = calculateMetrics(reference, observed, matches, anchors, line);
  const eligibleNotes = alignedReference.filter((note) => note.trainingEligible).length;
  const metrics = {
    ...baseMetrics,
    manualAnchorCount: manualAnchors.length,
    automaticAnchorCount: anchors.filter((anchor) => String(anchor.kind).startsWith('automatic')).length,
    qualityWindowCount: qualityWindows.length,
    trustedWindowCount: qualityWindows.filter((window) => window.status === 'trusted').length,
    manualAcceptedWindowCount: qualityWindows.filter((window) => window.status === 'accepted-manually').length,
    reviewWindowCount: qualityWindows.filter((window) => window.status === 'review').length,
    unsafeWindowCount: qualityWindows.filter((window) => window.status === 'unsafe').length,
    rejectedWindowCount: qualityWindows.filter((window) => window.status === 'rejected').length,
    unsafeOrRejectedWindowCount: qualityWindows.filter((window) => (
      window.status === 'unsafe' || window.status === 'rejected'
    )).length,
    trainingEligibleNotes: eligibleNotes,
    trainingEligiblePercent: Number((eligibleNotes / Math.max(1, alignedReference.length) * 100).toFixed(2)),
    sourceDurationSeconds: Number.isFinite(sourceDurationSeconds)
      ? Number(sourceDurationSeconds.toFixed(4))
      : null,
  };
  const supervisionPackage = {
    schema: 'polymath-supervision-package-v1',
    generatedAt: new Date().toISOString(),
    timeline: {
      sourceDurationSeconds: metrics.sourceDurationSeconds,
      referenceDurationSeconds: Number(maximumReferenceTime.toFixed(4)),
      rule: 'Reference MIDI notes are warped onto the source audio/video timeline; the source duration is never stretched to the MIDI duration.',
    },
    alignment: {
      metrics,
      coarse: line,
      anchors,
      manualAnchors,
      tempoSegments,
      qualityWindows,
    },
    review: {
      decisions: options.reviewDecisions || {},
      excludedRanges: options.excludedRanges || [],
      readyForTraining: metrics.unsafeWindowCount === 0
        && metrics.reviewWindowCount === 0
        && metrics.trainingEligibleNotes >= 8,
    },
    notes: alignedReference,
  };
  return {
    reference,
    observed,
    coarse: line,
    matches,
    automaticAnchors,
    manualAnchors,
    anchors,
    tempoSegments,
    qualityWindows,
    metrics,
    alignedReference,
    supervisionPackage,
  };
}

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

export function createAlignmentSvg(result, { width = 1200, height = 760 } = {}) {
  const margin = { left: 82, right: 34, top: 72, bottom: 82 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maxReference = Math.max(1, result.reference.at(-1)?.time ?? 1);
  const observedTimes = result.observed.map((note) => note.time);
  const maxObserved = Math.max(1, ...observedTimes, ...result.anchors.map((anchor) => anchor.observedTime));
  const x = (time) => margin.left + clamp(time / maxReference, 0, 1) * plotWidth;
  const y = (time) => margin.top + plotHeight - clamp(time / maxObserved, 0, 1) * plotHeight;
  const ticks = 8;
  const grid = Array.from({ length: ticks + 1 }, (_, index) => {
    const xTime = maxReference * index / ticks;
    const yTime = maxObserved * index / ticks;
    return `
      <line x1="${x(xTime)}" y1="${margin.top}" x2="${x(xTime)}" y2="${margin.top + plotHeight}" class="grid"/>
      <text x="${x(xTime)}" y="${margin.top + plotHeight + 28}" class="tick" text-anchor="middle">${xTime.toFixed(0)}s</text>
      <line x1="${margin.left}" y1="${y(yTime)}" x2="${margin.left + plotWidth}" y2="${y(yTime)}" class="grid"/>
      <text x="${margin.left - 12}" y="${y(yTime) + 4}" class="tick" text-anchor="end">${yTime.toFixed(0)}s</text>`;
  }).join('');
  const coarseStart = result.coarse.offset;
  const coarseEnd = maxReference * result.coarse.scale + result.coarse.offset;
  const warpPath = result.anchors.map((anchor, index) => (
    `${index ? 'L' : 'M'} ${x(anchor.referenceTime).toFixed(2)} ${y(anchor.observedTime).toFixed(2)}`
  )).join(' ');
  const points = result.matches.map((match) => {
    const fill = match.exactPitch ? '#2dd4bf' : '#f59e0b';
    const radius = match.exactPitch ? 3.2 : 4;
    const title = `${match.reference.note} @ ${match.reference.time.toFixed(3)}s → ${match.observed.note} @ ${match.observed.time.toFixed(3)}s`;
    return `<circle cx="${x(match.reference.time).toFixed(2)}" cy="${y(match.observed.time).toFixed(2)}" r="${radius}" fill="${fill}" opacity="0.78"><title>${escapeXml(title)}</title></circle>`;
  }).join('');
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    .background { fill: #101126; }
    .panel { fill: #181a38; stroke: #343868; }
    .grid { stroke: #2c3057; stroke-width: 1; }
    .tick { fill: #aeb4d8; font: 13px system-ui, sans-serif; }
    .label { fill: #eef0ff; font: 600 15px system-ui, sans-serif; }
    .title { fill: #ffffff; font: 700 24px system-ui, sans-serif; }
    .subtitle { fill: #bac0e8; font: 14px system-ui, sans-serif; }
  </style>
  <rect class="background" width="100%" height="100%"/>
  <text x="${margin.left}" y="34" class="title">Polymath note-coordinate alignment</text>
  <text x="${margin.left}" y="56" class="subtitle">Green = exact MIDI pitch · Amber = same pitch class, different octave · Purple = nonlinear time map</text>
  <rect class="panel" x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" rx="8"/>
  ${grid}
  <line x1="${x(0)}" y1="${y(coarseStart)}" x2="${x(maxReference)}" y2="${y(coarseEnd)}" stroke="#7c83ff" stroke-width="2" stroke-dasharray="8 7" opacity="0.7"/>
  <path d="${warpPath}" fill="none" stroke="#d65cff" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>
  ${points}
  <text x="${margin.left + plotWidth / 2}" y="${height - 25}" class="label" text-anchor="middle">Desired MIDI time (seconds)</text>
  <text x="22" y="${margin.top + plotHeight / 2}" class="label" text-anchor="middle" transform="rotate(-90 22 ${margin.top + plotHeight / 2})">MuScriptor / source-audio time (seconds)</text>
  <text x="${width - 38}" y="34" class="subtitle" text-anchor="end">Confidence ${(result.metrics.confidence * 100).toFixed(1)}% · ${escapeXml(result.metrics.verdict)}</text>
</svg>`;
}

function parseArguments(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) args[key] = true;
    else {
      args[key] = value;
      index += 1;
    }
  }
  return args;
}

function extractJsonNotes(payload) {
  const candidates = [payload?.notes, payload?.song?.notes, payload?.result?.notes, payload?.output?.notes];
  return candidates.find(Array.isArray) || [];
}

async function loadMidiNotes(filename) {
  const bytes = await fs.readFile(filename);
  const midi = new Midi(bytes);
  return midi.tracks.flatMap((track, trackIndex) => {
    if (track.channel === 9) return [];
    const instrument = track.instrument?.name || track.name || `track-${trackIndex + 1}`;
    return track.notes.map((note) => ({
      midi: note.midi,
      time: note.time,
      duration: note.duration,
      velocity: note.velocity,
      instrument,
    }));
  });
}

export async function loadNoteFile(filename) {
  const resolved = path.resolve(filename);
  const extension = path.extname(resolved).toLowerCase();
  if (extension === '.mid' || extension === '.midi') {
    return loadMidiNotes(resolved);
  }
  if (extension === '.json') {
    const payload = JSON.parse(await fs.readFile(resolved, 'utf8'));
    const notes = extractJsonNotes(payload);
    if (!notes.length) throw new Error(`No note array was found in ${path.basename(resolved)}.`);
    return notes;
  }
  throw new Error(`Unsupported note file: ${path.basename(resolved)}. Use MIDI or JSON.`);
}

async function runCli() {
  const args = parseArguments(process.argv.slice(2));
  const referenceFile = args.reference || args.midi;
  const observedFile = args.observed || args.muscriptor;
  if (!referenceFile || !observedFile) {
    throw new Error('Usage: npm run align:notes -- --reference ideal.mid-or-json --observed model.mid-or-json [--out alignment-output]');
  }
  const [referenceNotes, observedNotes] = await Promise.all([
    loadNoteFile(referenceFile),
    loadNoteFile(observedFile),
  ]);
  const result = alignNoteCoordinates(referenceNotes, observedNotes, {
    sourceDurationSeconds: args['source-duration'] || null,
  });
  const outputDirectory = path.resolve(args.out || 'alignment-output');
  await fs.mkdir(outputDirectory, { recursive: true });
  await Promise.all([
    fs.writeFile(
      path.join(outputDirectory, 'alignment-report.json'),
      `${JSON.stringify({
        metrics: result.metrics,
        coarse: result.coarse,
        anchors: result.anchors,
        matches: result.matches.map((match) => ({
          reference: match.reference,
          observed: match.observed,
          exactPitch: match.exactPitch,
          octaveDifference: match.octaveDifference,
          coarseResidual: match.coarseResidual,
        })),
      }, null, 2)}\n`,
    ),
    fs.writeFile(
      path.join(outputDirectory, 'aligned-training-labels.json'),
      `${JSON.stringify(result.supervisionPackage, null, 2)}\n`,
    ),
    fs.writeFile(path.join(outputDirectory, 'alignment-plot.svg'), createAlignmentSvg(result)),
  ]);
  process.stdout.write(`${JSON.stringify({
    referenceFile: path.resolve(referenceFile),
    observedFile: path.resolve(observedFile),
    outputDirectory,
    ...result.metrics,
  }, null, 2)}\n`);
}

const isDirectRun = process.argv[1]
  && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isDirectRun) {
  runCli().catch((error) => {
    process.stderr.write(`Alignment failed: ${error.message}\n`);
    process.exitCode = 1;
  });
}
