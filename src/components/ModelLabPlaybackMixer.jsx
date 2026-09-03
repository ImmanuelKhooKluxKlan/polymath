import { useEffect, useMemo, useRef, useState } from 'react';
import { pianoAudio } from '../engine/audioEngine.js';
import { guitarAudio } from '../engine/guitarEngine.js';
import { ensembleAudio } from '../engine/ensembleEngine.js';
import { INSTRUMENTS, INSTRUMENT_BY_ID } from '../data/instruments.js';

const OPEN_GUITAR_MIDI = [40, 45, 50, 55, 59, 64];
const LOOKAHEAD_SECONDS = 0.22;
const SCHEDULER_INTERVAL_MS = 25;
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function midiToNote(midi) {
  const value = Math.round(Number(midi));
  return `${NOTE_NAMES[((value % 12) + 12) % 12]}${Math.floor(value / 12) - 1}`;
}

function displayName(value) {
  return String(value || 'unknown')
    .replaceAll('_', ' ')
    .replaceAll('-', ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function naturalSoundFor(detected) {
  const value = String(detected || '').toLowerCase();
  if (value.includes('piano') || value.includes('keyboard')) return 'piano';
  if (value.includes('chromatic') && value.includes('percussion')) return 'synth';
  if (value.includes('organ')) return 'synth';
  if (value.includes('electric') && value.includes('guitar')) return 'electric-guitar';
  if (value.includes('guitar')) return 'guitar';
  if (value.includes('bass')) return 'upright-bass';
  if (value.includes('drum')) return 'drums';
  if (value.includes('cello')) return 'cello';
  if (value.includes('violin') || value.includes('fiddle') || value.includes('string')) return 'violin';
  if (value.includes('flute')) return 'flute';
  if (value.includes('sax')) return 'saxophone';
  if (value.includes('reed')) return 'saxophone';
  if (value.includes('trumpet') || value.includes('brass')) return 'trumpet';
  if (value.includes('pipe')) return 'flute';
  if (value.includes('clarinet')) return 'clarinet';
  if (value.includes('banjo')) return 'banjo';
  if (value.includes('mandolin')) return 'mandolin';
  if (value.includes('dobro') || value.includes('resonator')) return 'dobro';
  if (value.includes('ukulele')) return 'ukulele';
  if (value.includes('voice') || value.includes('vocal') || value.includes('choir')) return 'synth';
  if (value.includes('synth') || value.includes('percussive') || value.includes('effect')) return 'synth';
  return 'synth';
}

function secondsLabel(value) {
  const safe = Math.max(0, Number(value) || 0);
  return `${Math.floor(safe / 60)}:${String(Math.floor(safe % 60)).padStart(2, '0')}`;
}

function guitarPosition(midi) {
  return OPEN_GUITAR_MIDI
    .map((openMidi, stringIndex) => ({ stringIndex, fret: midi - openMidi }))
    .filter(({ fret }) => Number.isInteger(fret) && fret >= 0 && fret <= 24)
    .sort((a, b) => a.fret - b.fret || b.stringIndex - a.stringIndex)[0] || null;
}

function normalizeNotes(raw) {
  return (Array.isArray(raw?.notes) ? raw.notes : [])
    .map((note, index) => {
      const midi = Math.round(Number(note.midi));
      return {
        ...note,
        id: `${note.instrument || 'unknown'}-${midi}-${note.time}-${index}`,
        instrument: String(note.instrument || 'unknown').toLowerCase(),
        midi,
        note: note.note || midiToNote(midi),
        time: Math.max(0, Number(note.time) || 0),
        duration: Math.max(0.04, Number(note.duration) || 0.4),
        velocity: Math.max(0.08, Math.min(1, Number(note.velocity) || 0.76)),
      };
    })
    .filter((note) => Number.isFinite(note.midi) && note.midi >= 0 && note.midi <= 127)
    .sort((a, b) => a.time - b.time || a.midi - b.midi);
}

function selectionDiagnostics(notes) {
  const groups = new Map();
  notes.forEach((note) => {
    const key = `${note.instrument}:${note.midi}`;
    const group = groups.get(key) || [];
    group.push(note);
    groups.set(key, group);
  });
  let rapid = 0;
  let overlaps = 0;
  groups.forEach((group) => {
    group.sort((a, b) => a.time - b.time || b.duration - a.duration);
    for (let index = 1; index < group.length; index += 1) {
      const previous = group[index - 1];
      const current = group[index];
      if (current.time - previous.time <= 0.075) rapid += 1;
      if (current.time < previous.time + previous.duration) overlaps += 1;
    }
  });
  return { rapid, overlaps };
}

export default function ModelLabPlaybackMixer({ raw, instrumentStats = [], initialSound = 'piano' }) {
  const notes = useMemo(() => normalizeNotes(raw), [raw]);
  const detectedParts = useMemo(() => (
    [...new Set(notes.map((note) => note.instrument))]
      .sort((a, b) => a.localeCompare(b))
  ), [notes]);
  const duration = useMemo(() => Math.max(0, ...notes.map((note) => note.time + note.duration)), [notes]);
  const [selectedParts, setSelectedParts] = useState(() => new Set(detectedParts));
  const [selectedOwnedSound, setSelectedOwnedSound] = useState(
    INSTRUMENT_BY_ID[initialSound] ? initialSound : 'piano',
  );
  const [bandLayers, setBandLayers] = useState(() => new Set());
  const [soloSound, setSoloSound] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioStatus, setAudioStatus] = useState('Ready');
  const [skippedNotes, setSkippedNotes] = useState(0);
  const scheduler = useRef(null);
  const animation = useRef(null);
  const cursor = useRef(0);
  const startedAt = useRef(0);
  const scheduledNotes = useRef([]);
  const alive = useRef(true);
  const playbackGeneration = useRef(0);

  const chosenNotes = useMemo(() => (
    notes.filter((note) => selectedParts.has(note.instrument))
  ), [notes, selectedParts]);
  const diagnostics = useMemo(() => selectionDiagnostics(chosenNotes), [chosenNotes]);
  const layerCount = soloSound ? 1 : 1 + bandLayers.size;

  function clearTimers() {
    if (scheduler.current) window.clearInterval(scheduler.current);
    if (animation.current) window.cancelAnimationFrame(animation.current);
    scheduler.current = null;
    animation.current = null;
  }

  function silenceAll() {
    pianoAudio.stopAll({ releaseSeconds: 0.035 });
    guitarAudio.stopAll(0.035);
    ensembleAudio.stopAll(0.035);
  }

  function pausePlayback(reset = false) {
    playbackGeneration.current += 1;
    clearTimers();
    silenceAll();
    setIsPlaying(false);
    if (reset) setCurrentTime(0);
  }

  useEffect(() => {
    const startingSound = INSTRUMENT_BY_ID[initialSound] ? initialSound : 'piano';
    alive.current = true;
    setSelectedParts(new Set(detectedParts));
    setBandLayers(new Set());
    setSelectedOwnedSound(startingSound);
    setSoloSound(startingSound);
    setCurrentTime(0);
    setSkippedNotes(0);
    return () => {
      alive.current = false;
      clearTimers();
      silenceAll();
    };
  }, [raw, detectedParts, initialSound]);

  function togglePart(part) {
    pausePlayback(false);
    setSelectedParts((current) => {
      const next = new Set(current);
      if (next.has(part)) next.delete(part);
      else next.add(part);
      return next;
    });
  }

  function toggleBandLayer(sound) {
    pausePlayback(false);
    setSoloSound(null);
    setBandLayers((current) => {
      const next = new Set(current);
      if (next.has(sound)) next.delete(sound);
      else next.add(sound);
      return next;
    });
  }

  function soundTargetsFor(note, activeSoloSound = soloSound, activeBandLayers = bandLayers) {
    if (activeSoloSound) return [activeSoloSound];
    return [...new Set([naturalSoundFor(note.instrument), ...activeBandLayers])];
  }

  function playWithSound(note, sound, delaySeconds = 0) {
    const delay = Math.max(0, delaySeconds);
    if (sound === 'piano') {
      if (note.midi < 21 || note.midi > 108) return false;
      pianoAudio.playAt(
        note.note,
        note.velocity,
        note.duration,
        pianoAudio.getCurrentTime() + delay,
        { source: 'autoplay', retriggerSameNote: true },
      );
      return true;
    }
    if (sound === 'guitar') {
      const position = guitarPosition(note.midi);
      if (!position) return false;
      guitarAudio.playEvent({
        ...position,
        duration: note.duration,
        velocity: note.velocity,
        releaseSeconds: Math.min(0.55, Math.max(0.22, note.duration * 0.35)),
      }, guitarAudio.getCurrentTime() + delay, 1);
      return true;
    }
    ensembleAudio.playAt(
      note.note,
      sound,
      note.velocity,
      note.duration,
      ensembleAudio.getCurrentTime() + delay,
    );
    return true;
  }

  async function prepareSounds(targetSounds, playbackNotes) {
    targetSounds.forEach((sound) => {
      if (sound === 'piano') pianoAudio.ensure();
      else if (sound === 'guitar') guitarAudio.ensure();
      else ensembleAudio.ensure();
    });
    const preparations = targetSounds.map((sound) => {
      if (sound === 'piano') return pianoAudio.preloadSongNotes({ notes: playbackNotes });
      if (sound === 'guitar') return guitarAudio.preloadSamples();
      return ensembleAudio.preloadInstrument(sound);
    });
    await Promise.all(preparations);
  }

  function finishNaturally() {
    clearTimers();
    setCurrentTime(duration);
    setIsPlaying(false);
    setAudioStatus('Playback complete');
  }

  async function startPlayback({
    forcedSoloSound = soloSound,
    forcedBandLayers = bandLayers,
    restart = false,
  } = {}) {
    if (!chosenNotes.length || (isPlaying && !restart)) return;
    if (restart) pausePlayback(false);
    const generation = playbackGeneration.current + 1;
    playbackGeneration.current = generation;
    const position = currentTime >= duration - 0.02 ? 0 : currentTime;
    const playbackNotes = chosenNotes.filter((note) => note.time >= position - 0.001);
    const targetSounds = [...new Set(playbackNotes.flatMap((note) => (
      soundTargetsFor(note, forcedSoloSound, forcedBandLayers)
    )))];
    setAudioStatus(`Preparing ${targetSounds.length} Polymath sound${targetSounds.length === 1 ? '' : 's'}…`);
    setSkippedNotes(0);
    silenceAll();
    try {
      await prepareSounds(targetSounds, playbackNotes);
    } catch {
      // Every engine has a synthesis fallback when a recording pack is unavailable.
    }
    if (!alive.current || generation !== playbackGeneration.current) return;

    scheduledNotes.current = playbackNotes;
    cursor.current = 0;
    startedAt.current = performance.now() - position * 1000;
    setCurrentTime(position);
    setIsPlaying(true);
    setAudioStatus(forcedSoloSound
      ? `Playing every selected note as ${INSTRUMENT_BY_ID[forcedSoloSound]?.label || forcedSoloSound}`
      : forcedBandLayers.size
        ? `Natural mix + ${forcedBandLayers.size} added band layer${forcedBandLayers.size === 1 ? '' : 's'}`
        : 'Natural detected-instrument mix');

    const schedule = () => {
      const now = (performance.now() - startedAt.current) / 1000;
      const lookAhead = now + LOOKAHEAD_SECONDS;
      let skipped = 0;
      while (
        cursor.current < scheduledNotes.current.length
        && scheduledNotes.current[cursor.current].time <= lookAhead
      ) {
        const note = scheduledNotes.current[cursor.current];
        const delay = Math.max(0, note.time - now);
        soundTargetsFor(note, forcedSoloSound, forcedBandLayers).forEach((sound) => {
          if (!playWithSound(note, sound, delay)) skipped += 1;
        });
        cursor.current += 1;
      }
      if (skipped) setSkippedNotes((value) => value + skipped);
      if (now >= duration) finishNaturally();
    };

    const animate = () => {
      const now = Math.min(duration, (performance.now() - startedAt.current) / 1000);
      setCurrentTime(now);
      if (now < duration) animation.current = window.requestAnimationFrame(animate);
    };

    schedule();
    scheduler.current = window.setInterval(schedule, SCHEDULER_INTERVAL_MS);
    animation.current = window.requestAnimationFrame(animate);
  }

  async function previewOwnedSound() {
    pausePlayback(false);
    const instrument = INSTRUMENT_BY_ID[selectedOwnedSound];
    const previewNote = instrument?.manualNotes?.[0] || (selectedOwnedSound === 'drums' ? 'C2' : 'C4');
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const match = previewNote.match(/^([A-G])(#|b)?(-?\d+)$/);
    const aliases = { Db: 'C#', Eb: 'D#', Gb: 'F#', Ab: 'G#', Bb: 'A#' };
    const pitchClass = match ? aliases[`${match[1]}${match[2] || ''}`] || `${match[1]}${match[2] || ''}` : 'C';
    const midi = match ? (Number(match[3]) + 1) * 12 + noteNames.indexOf(pitchClass) : 60;
    const preview = { note: previewNote, midi, duration: 0.9, velocity: 0.82, instrument: 'preview' };
    setAudioStatus(`Previewing ${instrument?.label || selectedOwnedSound}`);
    await prepareSounds([selectedOwnedSound], [preview]).catch(() => {});
    playWithSound(preview, selectedOwnedSound, 0.01);
  }

  const statsByInstrument = useMemo(() => Object.fromEntries(
    instrumentStats.map((item) => [item.instrument, item]),
  ), [instrumentStats]);

  return (
    <section className="model-lab-panel model-lab-playback-panel">
      <div className="model-lab-panel-heading">
        <div>
          <p className="eyebrow">A/B playback diagnosis</p>
          <h2>Hear where the stutter comes from</h2>
          <p>Select detected parts on the left, then hear this model’s notes through any Polymath-owned instrument on the right.</p>
        </div>
        <div className={`model-lab-play-state ${diagnostics.rapid ? 'warning' : ''}`}>
          <span>{audioStatus}</span>
          <strong>{diagnostics.rapid} rapid repeats in selection</strong>
        </div>
      </div>

      <div className="model-lab-mixer-columns">
        <article className="model-lab-mixer-column">
          <div className="model-lab-mixer-column-heading">
            <div><span>1</span><h3>Detected parts</h3></div>
            <div>
              <button type="button" onClick={() => { pausePlayback(false); setSelectedParts(new Set(detectedParts)); }}>All</button>
              <button type="button" onClick={() => { pausePlayback(false); setSelectedParts(new Set()); }}>None</button>
            </div>
          </div>
          <div className="model-lab-part-list">
            {detectedParts.map((part) => {
              const mapped = naturalSoundFor(part);
              const stat = statsByInstrument[part];
              return (
                <label key={part} className={selectedParts.has(part) ? 'active' : ''}>
                  <input type="checkbox" checked={selectedParts.has(part)} onChange={() => togglePart(part)} />
                  <span>
                    <strong>{displayName(part)}</strong>
                    <small>{stat?.notes || 0} notes · {stat?.minimumNote || '?'}–{stat?.maximumNote || '?'}</small>
                    <em>Natural sound: {INSTRUMENT_BY_ID[mapped]?.label || mapped}</em>
                  </span>
                </label>
              );
            })}
          </div>
        </article>

        <article className="model-lab-mixer-column">
          <div className="model-lab-mixer-column-heading">
            <div><span>2</span><h3>Polymath-owned sounds</h3></div>
          </div>
          <label className="model-lab-sound-select">
            <span>Choose an owned instrument</span>
            <select value={selectedOwnedSound} onChange={(event) => setSelectedOwnedSound(event.target.value)}>
              {INSTRUMENTS.map((instrument) => (
                <option key={instrument.id} value={instrument.id}>{instrument.label}</option>
              ))}
            </select>
          </label>
          <div className="model-lab-sound-actions">
            <button type="button" className="ghost" onClick={previewOwnedSound}>Preview one note</button>
            <button
              type="button"
              className={soloSound === selectedOwnedSound ? 'primary' : 'ghost'}
              disabled={!chosenNotes.length}
              onClick={async () => {
                setSoloSound(selectedOwnedSound);
                setBandLayers(new Set());
                await startPlayback({
                  forcedSoloSound: selectedOwnedSound,
                  forcedBandLayers: new Set(),
                  restart: true,
                });
              }}
            >
              Play selection as {INSTRUMENT_BY_ID[selectedOwnedSound]?.label || selectedOwnedSound}
            </button>
            <button
              type="button"
              className={bandLayers.has(selectedOwnedSound) ? 'primary' : 'ghost'}
              onClick={() => toggleBandLayer(selectedOwnedSound)}
            >
              {bandLayers.has(selectedOwnedSound) ? 'Remove from band' : 'Add into band'}
            </button>
            <button
              type="button"
              className={!soloSound && !bandLayers.size ? 'primary' : 'ghost'}
              onClick={() => {
                pausePlayback(false);
                setSoloSound(null);
                setBandLayers(new Set());
              }}
            >
              Natural detected mix
            </button>
          </div>
          <div className="model-lab-band-layers">
            <span>Current rendering</span>
            {soloSound ? (
              <strong>Solo: {INSTRUMENT_BY_ID[soloSound]?.label}</strong>
            ) : (
              <>
                <strong>Natural detected sounds</strong>
                {[...bandLayers].map((sound) => (
                  <button type="button" key={sound} onClick={() => toggleBandLayer(sound)}>
                    + {INSTRUMENT_BY_ID[sound]?.label} ×
                  </button>
                ))}
              </>
            )}
          </div>
        </article>
      </div>

      <div className="model-lab-transport">
        <button type="button" className="primary" disabled={!chosenNotes.length} onClick={() => (isPlaying ? pausePlayback(false) : startPlayback())}>
          {isPlaying ? 'Pause' : 'Play selected test'}
        </button>
        <button type="button" className="ghost" onClick={() => pausePlayback(true)}>Stop</button>
        <span>{secondsLabel(currentTime)}</span>
        <input
          type="range"
          min="0"
          max={Math.max(duration, 0.01)}
          step="0.01"
          value={Math.min(currentTime, duration)}
          onChange={(event) => {
            pausePlayback(false);
            setCurrentTime(Number(event.target.value));
          }}
          aria-label="Model Lab playback position"
        />
        <span>{secondsLabel(duration)}</span>
      </div>

      <div className="model-lab-live-diagnostics">
        <span><strong>{chosenNotes.length}</strong> selected notes</span>
        <span><strong>{selectedParts.size}</strong> detected parts</span>
        <span><strong>{layerCount}</strong> render layer{layerCount === 1 ? '' : 's'}</span>
        <span className={diagnostics.rapid ? 'warning' : ''}><strong>{diagnostics.rapid}</strong> ≤75 ms repeats</span>
        <span className={diagnostics.overlaps ? 'warning' : ''}><strong>{diagnostics.overlaps}</strong> same-pitch overlaps</span>
        <span className={skippedNotes ? 'warning' : ''}><strong>{skippedNotes}</strong> out-of-range skips</span>
      </div>
    </section>
  );
}

export { naturalSoundFor, selectionDiagnostics };
