const INSTRUMENT_TRIMS = {
  piano: 0.76,
  guitar: 0.82,
  'electric-guitar': 0.70,
  fiddle: 0.68,
  violin: 0.68,
  cello: 0.82,
  banjo: 0.62,
  mandolin: 0.68,
  dobro: 0.72,
  'upright-bass': 0.94,
  'bass-guitar': 0.90,
  ukulele: 0.78,
  drums: 0.72,
  synth: 0.66,
  flute: 0.72,
  saxophone: 0.70,
  trumpet: 0.66,
  clarinet: 0.74,
};

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function percentile(values, amount) {
  if (!values.length) return 0;
  const sorted = [...values].sort((first, second) => first - second);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * amount))];
}

export function analyzeScoreLoudness(score) {
  const notes = Array.isArray(score?.notes) ? score.notes : [];
  if (!notes.length) {
    return {
      noteCount: 0,
      duration: 0,
      averageVelocity: 0.8,
      velocityPeak: 0.8,
      notesPerSecond: 0,
      peakPolyphony: 1,
      sourceConfidence: 1,
      sourceLoudnessDb: null,
    };
  }

  const duration = Math.max(0.1, ...notes.map((note) => (
    Math.max(0, Number(note.time) || 0) + Math.max(0.04, Number(note.duration) || 0.4)
  )));
  const velocities = notes.map((note) => clamp(Number(note.velocity) || 0.8, 0.05, 1.2));
  const events = [];
  notes.forEach((note) => {
    const start = Math.max(0, Number(note.time) || 0);
    const end = start + Math.max(0.04, Number(note.duration) || 0.4);
    events.push([start, 1], [end, -1]);
  });
  events.sort((first, second) => first[0] - second[0] || first[1] - second[1]);
  let active = 0;
  let peakPolyphony = 1;
  events.forEach(([, change]) => {
    active += change;
    peakPolyphony = Math.max(peakPolyphony, active);
  });

  return {
    noteCount: notes.length,
    duration,
    averageVelocity: velocities.reduce((sum, value) => sum + value, 0) / velocities.length,
    velocityPeak: percentile(velocities, 0.95),
    notesPerSecond: notes.length / duration,
    peakPolyphony,
    sourceConfidence: clamp(Number(score?.transcription?.noteConfidence) || 1, 0.2, 1),
    sourceLoudnessDb: Number.isFinite(Number(score?.transcription?.sourceAudio?.loudnessDb))
      ? Number(score.transcription.sourceAudio.loudnessDb)
      : null,
  };
}

export function estimateBandLoudness(band) {
  const activeParts = (band?.instruments || []).filter((part) => {
    const score = part.score || band?.generalScore;
    return !part.muted && Array.isArray(score?.notes) && score.notes.length;
  });
  if (!activeParts.length) return [];

  const estimates = activeParts.map((part) => {
    const score = part.score || band.generalScore;
    const metrics = analyzeScoreLoudness(score);
    const instrumentTrim = INSTRUMENT_TRIMS[part.instrument] ?? 0.76;
    const velocityCorrection = Math.pow(0.8 / clamp(metrics.averageVelocity, 0.3, 1.1), 0.5);
    const densityCorrection = Math.pow(2.2 / clamp(metrics.notesPerSecond, 0.35, 8), 0.12);
    const polyphonyCorrection = Math.pow(1 / clamp(metrics.peakPolyphony, 1, 8), 0.16);
    const confidenceHeadroom = 0.88 + metrics.sourceConfidence * 0.12;
    const sourceLevelCorrection = metrics.sourceLoudnessDb === null
      ? 1
      : clamp(Math.pow(10, (-18 - metrics.sourceLoudnessDb) / 40), 0.86, 1.12);
    const volume = instrumentTrim
      * velocityCorrection
      * densityCorrection
      * polyphonyCorrection
      * confidenceHeadroom
      * sourceLevelCorrection;
    return { part, score, metrics, volume };
  });

  const estimatedPeak = estimates.reduce((sum, estimate) => (
    sum + estimate.volume * estimate.metrics.velocityPeak
      * Math.sqrt(clamp(estimate.metrics.peakPolyphony, 1, 8))
  ), 0);
  const mixHeadroom = Math.min(1, 2.5 / Math.max(1, estimatedPeak));

  return estimates.map(({ part, score, metrics, volume }) => {
    const finalVolume = clamp(volume * mixHeadroom, 0.34, 1.08);
    const reasons = [
      `${part.instrument} output calibrated`,
      metrics.notesPerSecond > 3.5 ? 'dense part reduced' : metrics.notesPerSecond < 1 ? 'sparse part supported' : 'moderate note density',
      metrics.peakPolyphony > 1 ? `${metrics.peakPolyphony}-note peak overlap controlled` : 'monophonic headroom',
    ];
    if (score?.transcription) reasons.push('recording level and confidence considered');
    if (mixHeadroom < 0.98) reasons.push('whole-band clipping protection');
    return {
      partId: part.id,
      volume: Number(finalVolume.toFixed(2)),
      reasons,
      metrics,
    };
  });
}
