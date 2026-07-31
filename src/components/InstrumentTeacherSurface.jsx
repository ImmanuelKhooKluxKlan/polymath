import { midiToNote, parseNote } from '../engine/noteMath.js';

const STRING_LAYOUTS = {
  guitar: {
    tuning: ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'],
    maxPosition: 15,
    body: 'guitar',
    action: 'Fret and pick',
  },
  fiddle: {
    tuning: ['C3', 'G3', 'D4', 'A4', 'E5'],
    maxPosition: 12,
    body: 'fiddle',
    action: 'Place your finger',
    fretless: true,
  },
  violin: {
    tuning: ['G3', 'D4', 'A4', 'E5'],
    maxPosition: 12,
    body: 'fiddle',
    action: 'Place your finger and bow',
    fretless: true,
  },
  cello: {
    tuning: ['C2', 'G2', 'D3', 'A3'],
    maxPosition: 18,
    body: 'fiddle',
    action: 'Stop the string and bow',
    fretless: true,
  },
  banjo: {
    tuning: ['G4', 'D3', 'G3', 'B3', 'D4'],
    maxPosition: 12,
    body: 'banjo',
    action: 'Fret and pluck',
    droneStartFret: 5,
  },
  mandolin: {
    tuning: ['G3', 'D4', 'A4', 'E5'],
    maxPosition: 12,
    body: 'mandolin',
    action: 'Press the course',
    paired: true,
  },
  dobro: {
    tuning: ['G2', 'B2', 'D3', 'G3', 'B3', 'D4'],
    maxPosition: 12,
    body: 'dobro',
    action: 'Move the steel bar',
  },
  'upright-bass': {
    tuning: ['E1', 'A1', 'D2', 'G2'],
    maxPosition: 18,
    body: 'bass',
    action: 'Stop and pluck',
    fretless: true,
  },
  ukulele: {
    tuning: ['G4', 'C4', 'E4', 'A4'],
    maxPosition: 12,
    body: 'ukulele',
    action: 'Fret and strum',
  },
  'electric-guitar': {
    tuning: ['E2', 'A2', 'D3', 'G3', 'B3', 'E4'],
    maxPosition: 15,
    body: 'electric-guitar',
    action: 'Fret and pick',
  },
};

const DRUM_PARTS = [
  { id: 'crash', label: 'Crash', note: 'C#3', x: 170, y: 90, r: 58 },
  { id: 'hi-hat', label: 'Hi-hat', note: 'F#2', x: 300, y: 160, r: 52 },
  { id: 'tom-high', label: 'High tom', note: 'C3', x: 455, y: 150, r: 62 },
  { id: 'tom-mid', label: 'Mid tom', note: 'A2', x: 585, y: 165, r: 68 },
  { id: 'ride', label: 'Ride', note: 'D#3', x: 775, y: 105, r: 64 },
  { id: 'snare', label: 'Snare', note: 'D2', x: 350, y: 300, r: 78 },
  { id: 'floor-tom', label: 'Floor tom', note: 'G2', x: 690, y: 300, r: 88 },
  { id: 'kick', label: 'Kick', note: 'C2', x: 520, y: 330, r: 105 },
];

function safeMidi(note) {
  try {
    return parseNote(note).midi;
  } catch {
    return null;
  }
}

function mapNoteToString(note, layout) {
  const midi = safeMidi(note);
  if (midi == null) return null;
  const candidates = layout.tuning
    .map((openNote, stringIndex) => {
      const fret = midi - parseNote(openNote).midi;
      return { openNote, stringIndex, fret };
    })
    .filter((candidate) => {
      const positionOffset = layout.droneStartFret && candidate.stringIndex === 0 ? layout.droneStartFret : 0;
      return candidate.fret >= 0 && candidate.fret + positionOffset <= layout.maxPosition;
    })
    .sort((a, b) => a.fret - b.fret || b.stringIndex - a.stringIndex);
  return candidates[0] || null;
}

function horizontalPositionX(layout, stringIndex, fret, neckStart, fretWidth) {
  const positionOffset = layout.droneStartFret && stringIndex === 0 ? layout.droneStartFret : 0;
  const physicalPosition = fret + positionOffset;
  return physicalPosition === 0 ? 71 : neckStart + (physicalPosition - 0.5) * fretWidth;
}

function stringY(index, count) {
  if (count === 1) return 210;
  return 105 + (index * 210) / (count - 1);
}

function bodyArtwork(type) {
  if (type === 'banjo') {
    return (
      <g>
        <circle cx="820" cy="210" r="150" className="teacher-body teacher-body-light" />
        <circle cx="820" cy="210" r="116" className="teacher-resonator" />
        <circle cx="820" cy="210" r="34" className="teacher-sound-hole" />
      </g>
    );
  }
  if (type === 'electric-guitar') {
    return (
      <g>
        <path d="M735 92 C790 55 900 72 902 142 C905 188 866 190 910 240 C950 286 904 365 814 342 C760 328 716 302 698 260 C674 202 682 128 735 92Z" className="teacher-body teacher-electric" />
        <path d="M780 143 C815 124 851 128 874 151 C845 172 827 199 821 230 C786 213 766 179 780 143Z" className="teacher-pickguard" />
        <circle cx="850" cy="260" r="16" className="teacher-knob" />
        <circle cx="885" cy="280" r="14" className="teacher-knob" />
      </g>
    );
  }
  if (type === 'bass') {
    return (
      <g>
        <path d="M760 28 C815 30 844 82 820 127 C890 150 902 235 846 270 C868 330 826 398 758 390 C690 398 648 330 670 270 C614 235 626 150 696 127 C672 82 705 30 760 28Z" className="teacher-body teacher-bass-body" />
        <path d="M742 46 L778 46 L792 360 L728 360Z" className="teacher-fingerboard" />
        <path d="M690 202 C718 181 738 180 759 204 C780 180 801 181 828 202" className="teacher-bridge" />
      </g>
    );
  }
  if (type === 'fiddle') {
    return (
      <g>
        <path d="M742 65 C791 45 842 83 826 132 C882 145 895 214 850 245 C886 300 850 363 790 352 C758 346 747 324 734 303 C720 324 709 346 677 352 C617 363 581 300 617 245 C572 214 585 145 641 132 C625 83 676 45 725 65 C731 70 736 70 742 65Z" className="teacher-body teacher-fiddle-body" />
        <path d="M711 74 L756 74 L770 306 L697 306Z" className="teacher-fingerboard" />
        <path d="M648 205 Q684 178 707 206 M760 206 Q786 178 822 205" className="teacher-f-hole" />
      </g>
    );
  }
  const bodyClass = type === 'dobro' ? 'teacher-dobro-body' : 'teacher-acoustic-body';
  return (
    <g>
      <path d="M740 68 C800 45 867 85 847 148 C915 168 920 278 850 302 C865 358 799 391 742 348 C688 392 621 358 636 302 C566 278 571 168 639 148 C619 85 680 45 740 68Z" className={`teacher-body ${bodyClass}`} />
      {type === 'dobro' ? (
        <>
          <circle cx="752" cy="213" r="82" className="teacher-resonator" />
          <circle cx="752" cy="213" r="53" className="teacher-resonator-inner" />
        </>
      ) : (
        <circle cx="752" cy="213" r="49" className="teacher-sound-hole" />
      )}
    </g>
  );
}

function FrettedInstrument({ instrument, activeNotes, onPlay }) {
  const layout = STRING_LAYOUTS[instrument];
  const targets = [...activeNotes]
    .map((note) => ({ note, target: mapNoteToString(note, layout) }))
    .filter((item) => item.target);
  const fretCount = layout.maxPosition;
  const neckStart = 90;
  const neckEnd = 700;
  const fretWidth = (neckEnd - neckStart) / (fretCount + 1);
  const targetText = targets.length
    ? targets.map(({ note, target }) => `${note}: string ${target.stringIndex + 1}, ${target.fret === 0 ? 'open' : `position ${target.fret}`}`).join(' • ')
    : 'Press play, upload a ready-to-play sheet, or tap the strings to begin.';

  return (
    <div className="instrument-teacher-surface">
      <div className="teacher-callout" aria-live="polite">
        <span className="teacher-live-dot" />
        <strong>{layout.action}</strong>
        <span>{targetText}</span>
      </div>
      <svg viewBox="0 0 1000 420" role="img" aria-label={`Interactive ${instrument} teaching view`}>
        <defs>
          <linearGradient id={`wood-${instrument}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#f4b45f" />
            <stop offset="0.5" stopColor="#a85d32" />
            <stop offset="1" stopColor="#47241d" />
          </linearGradient>
          <filter id={`glow-${instrument}`} x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="10" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {bodyArtwork(layout.body)}
        <path d="M50 82 L108 70 L108 350 L50 338 C29 265 29 155 50 82Z" className="teacher-headstock" />
        <rect x={neckStart} y="83" width={neckEnd - neckStart + 85} height="254" rx="20" className="teacher-neck" />
        {Array.from({ length: fretCount + 1 }, (_, fret) => {
          const x = neckStart + fret * fretWidth;
          return <line key={`fret-${fret}`} x1={x} y1="83" x2={x} y2="337" className={fret === 0 ? 'teacher-nut' : layout.fretless ? 'teacher-position-guide' : 'teacher-fret'} />;
        })}
        {[3, 5, 7, 9, 12, 15].filter((fret) => fret <= fretCount).map((fret) => {
          const x = neckStart + (fret - 0.5) * fretWidth;
          return <circle key={`dot-${fret}`} cx={x} cy="210" r={fret === 12 ? 8 : 5} className="teacher-fret-dot" />;
        })}
        {layout.tuning.map((openNote, stringIndex) => {
          const y = stringY(stringIndex, layout.tuning.length);
          return (
            <g key={openNote + stringIndex}>
              {layout.paired ? (
                <>
                  <line x1="43" y1={y - 3} x2="905" y2={y - 3} className="teacher-string" style={{ strokeWidth: 1.4 + stringIndex * 0.45 }} />
                  <line x1="43" y1={y + 3} x2="905" y2={y + 3} className="teacher-string" style={{ strokeWidth: 1.4 + stringIndex * 0.45 }} />
                </>
              ) : (
                <line
                  x1={layout.droneStartFret && stringIndex === 0 ? neckStart + layout.droneStartFret * fretWidth : 43}
                  y1={y}
                  x2="905"
                  y2={y}
                  className="teacher-string"
                  style={{ strokeWidth: 1.4 + stringIndex * 0.45 }}
                />
              )}
              <text x="24" y={y + 5} className="teacher-string-label">{openNote}</text>
              {Array.from({ length: fretCount + 1 }, (_, fret) => {
                const midi = parseNote(openNote).midi + fret;
                const note = midiToNote(midi);
                const x = horizontalPositionX(layout, stringIndex, fret, neckStart, fretWidth);
                return (
                  <circle
                    key={`${stringIndex}-${fret}`}
                    cx={x}
                    cy={y}
                    r="14"
                    className="teacher-hit-area"
                    tabIndex="0"
                    role="button"
                    aria-label={`Play ${note}, string ${stringIndex + 1}, ${fret === 0 ? 'open' : `position ${fret}`}`}
                    onClick={() => onPlay(note)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') onPlay(note);
                    }}
                  />
                );
              })}
            </g>
          );
        })}
        {['fiddle', 'violin', 'cello'].includes(instrument) && (
          <g className="teacher-bow" aria-hidden="true">
            <line x1="545" y1="350" x2="915" y2="82" />
            <line x1="555" y1="362" x2="925" y2="94" className="teacher-bow-hair" />
            <text x="875" y="128">BOW</text>
          </g>
        )}
        {targets.map(({ note, target }, index) => {
          const y = stringY(target.stringIndex, layout.tuning.length);
          const x = horizontalPositionX(layout, target.stringIndex, target.fret, neckStart, fretWidth);
          return (
            <g key={`${note}-${index}`} className="teacher-target" filter={`url(#glow-${instrument})`}>
              {instrument === 'dobro' && <line x1={x} y1="86" x2={x} y2="334" className="teacher-slide-bar" />}
              <circle cx={x} cy={y} r="22" />
              <circle cx={x} cy={y} r="8" className="teacher-target-core" />
              <text x={x} y={y - 28} textAnchor="middle">{note}</text>
            </g>
          );
        })}
        <text x="500" y="392" textAnchor="middle" className="teacher-copy-caption">Watch the glow, place your hand on the same string/position, then copy the timing.</text>
      </svg>
    </div>
  );
}

function VerticalBassTeacher({ activeNotes, onPlay }) {
  const layout = STRING_LAYOUTS['upright-bass'];
  const targets = [...activeNotes]
    .map((note) => ({ note, target: mapNoteToString(note, layout) }))
    .filter((item) => item.target);
  const neckTop = 88;
  const positionStep = 14;
  const stringX = (index) => 420 + index * 54;
  const targetText = targets.length
    ? targets.map(({ note, target }) => `${note}: string ${target.stringIndex + 1}, position ${target.fret}`).join(' • ')
    : 'Tap a string position, or press play to follow the glowing left-hand target.';

  return (
    <div className="instrument-teacher-surface upright-teacher-surface">
      <div className="teacher-callout" aria-live="polite">
        <span className="teacher-live-dot" />
        <strong>Stop the string, then pluck</strong>
        <span>{targetText}</span>
      </div>
      <svg viewBox="0 0 1000 560" role="img" aria-label="Interactive upright double bass teaching view">
        <defs>
          <filter id="bass-glow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="11" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <path d="M455 22 Q500 2 545 22 L558 62 Q530 82 500 82 Q470 82 442 62Z" className="teacher-bass-scroll" />
        <path d="M454 70 L546 70 L566 354 L434 354Z" className="teacher-bass-neck-vertical" />
        <path d="M432 256 C350 234 296 303 328 368 C270 415 312 528 410 511 C446 505 474 480 500 452 C526 480 554 505 590 511 C688 528 730 415 672 368 C704 303 650 234 568 256 C548 268 525 282 500 300 C475 282 452 268 432 256Z" className="teacher-body teacher-bass-body" />
        <path d="M394 362 C423 340 454 347 475 373 M525 373 C546 347 577 340 606 362" className="teacher-f-hole" />
        <path d="M438 404 Q500 377 562 404" className="teacher-bridge" />
        <line x1="500" y1="510" x2="500" y2="548" className="teacher-endpin" />
        {Array.from({ length: layout.maxPosition + 1 }, (_, position) => {
          const y = neckTop + position * positionStep;
          return <line key={position} x1="438" y1={y} x2="562" y2={y} className={position === 0 ? 'teacher-nut-horizontal' : 'teacher-position-guide-horizontal'} />;
        })}
        {layout.tuning.map((openNote, stringIndex) => {
          const x = stringX(stringIndex);
          return (
            <g key={openNote}>
              <line x1={x} y1="58" x2={x} y2="487" className="teacher-string" style={{ strokeWidth: 1.8 + stringIndex * 0.7 }} />
              <text x={x} y="49" textAnchor="middle" className="teacher-string-label">{openNote}</text>
              {Array.from({ length: layout.maxPosition + 1 }, (_, position) => {
                const note = midiToNote(parseNote(openNote).midi + position);
                const y = neckTop + position * positionStep;
                return (
                  <circle
                    key={`${stringIndex}-${position}`}
                    cx={x}
                    cy={y}
                    r="15"
                    className="teacher-hit-area"
                    tabIndex="0"
                    role="button"
                    aria-label={`Play ${note}, string ${stringIndex + 1}, position ${position}`}
                    onClick={() => onPlay(note)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') onPlay(note);
                    }}
                  />
                );
              })}
            </g>
          );
        })}
        {targets.map(({ note, target }, index) => {
          const x = stringX(target.stringIndex);
          const y = neckTop + target.fret * positionStep;
          return (
            <g key={`${note}-${index}`} className="teacher-target" filter="url(#bass-glow)">
              <circle cx={x} cy={y} r="24" />
              <circle cx={x} cy={y} r="9" className="teacher-target-core" />
              <text x={x + 34} y={y + 7}>{note}</text>
            </g>
          );
        })}
        <g className="teacher-pluck-hand" aria-hidden="true">
          <path d="M626 350 Q690 330 706 386 Q678 422 630 406Z" />
          <text x="668" y="382" textAnchor="middle">PLUCK</text>
        </g>
        <text x="500" y="548" textAnchor="middle" className="teacher-copy-caption">Stand the bass upright, match the glowing string/position, and pluck beside the fingerboard.</text>
      </svg>
    </div>
  );
}

function DrumTeacher({ activeNotes, onPlay }) {
  const active = new Set([...activeNotes].map((note) => midiToNote(parseNote(note).midi)));
  return (
    <div className="instrument-teacher-surface">
      <div className="teacher-callout" aria-live="polite">
        <span className="teacher-live-dot" />
        <strong>Strike the glowing drum or cymbal</strong>
        <span>{active.size ? [...active].join(' + ') : 'Tap any drum piece, or press play to follow a beat.'}</span>
      </div>
      <svg viewBox="0 0 1000 470" role="img" aria-label="Interactive drum set teaching view">
        <defs>
          <filter id="drum-glow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="12" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <path d="M120 415 H880" className="teacher-drum-floor" />
        {DRUM_PARTS.map((part) => {
          const isActive = active.has(part.note);
          const isCymbal = ['crash', 'hi-hat', 'ride'].includes(part.id);
          return (
            <g
              key={part.id}
              className={`drum-piece ${isActive ? 'active' : ''}`}
              role="button"
              tabIndex="0"
              aria-label={`Strike ${part.label}`}
              onClick={() => onPlay(part.note)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') onPlay(part.note);
              }}
              filter={isActive ? 'url(#drum-glow)' : undefined}
            >
              {isCymbal ? (
                <>
                  <ellipse cx={part.x} cy={part.y} rx={part.r} ry={Math.max(16, part.r * 0.23)} className="teacher-cymbal" />
                  <line x1={part.x} y1={part.y + 8} x2={part.x} y2="416" className="teacher-stand" />
                </>
              ) : (
                <>
                  <circle cx={part.x} cy={part.y} r={part.r} className="teacher-drum" />
                  <circle cx={part.x} cy={part.y} r={part.r * 0.72} className="teacher-drum-head" />
                </>
              )}
              <text x={part.x} y={part.y + 5} textAnchor="middle" className="teacher-drum-label">{part.label}</text>
              <text x={part.x} y={part.y + 24} textAnchor="middle" className="teacher-drum-note">{part.note}</text>
            </g>
          );
        })}
        <text x="500" y="456" textAnchor="middle" className="teacher-copy-caption">Copy the highlighted piece: kick with foot, hi-hat with left hand, snare/toms/cymbals with sticks.</text>
      </svg>
    </div>
  );
}

function SynthTeacher({ activeNotes, onPlay }) {
  const startMidi = parseNote('C3').midi;
  const endMidi = parseNote('C6').midi;
  const notes = Array.from({ length: endMidi - startMidi + 1 }, (_, index) => midiToNote(startMidi + index));
  const whiteNotes = notes.filter((note) => !note.includes('#'));
  const whiteWidth = 900 / whiteNotes.length;
  const active = new Set([...activeNotes].map((note) => midiToNote(parseNote(note).midi)));
  return (
    <div className="instrument-teacher-surface">
      <div className="teacher-callout" aria-live="polite">
        <span className="teacher-live-dot" />
        <strong>Press the glowing synth keys</strong>
        <span>{active.size ? [...active].join(' + ') : 'Tap a key, or press play to follow the electronic-pop part.'}</span>
      </div>
      <svg viewBox="0 0 1000 420" role="img" aria-label="Interactive synth keyboard teaching view">
        <rect x="28" y="42" width="944" height="330" rx="34" className="teacher-synth-case" />
        <rect x="66" y="75" width="205" height="72" rx="13" className="teacher-synth-screen" />
        <g className="teacher-synth-controls">
          {Array.from({ length: 7 }, (_, i) => <circle key={i} cx={340 + i * 72} cy="108" r="19" />)}
        </g>
        <g transform="translate(50 175)">
          {whiteNotes.map((note, index) => {
            const isActive = active.has(note);
            return (
              <g key={note} onClick={() => onPlay(note)} className={`teacher-key-group ${isActive ? 'active' : ''}`}>
                <rect x={index * whiteWidth} y="0" width={whiteWidth - 2} height="188" rx="6" className="teacher-white-key" />
                <text x={index * whiteWidth + whiteWidth / 2} y="170" textAnchor="middle" className="teacher-key-label">{note}</text>
              </g>
            );
          })}
          {notes.filter((note) => note.includes('#')).map((note) => {
            const midi = parseNote(note).midi;
            const previousWhiteCount = notes.filter((candidate) => parseNote(candidate).midi < midi && !candidate.includes('#')).length;
            const x = previousWhiteCount * whiteWidth - whiteWidth * 0.32;
            const isActive = active.has(note);
            return (
              <g key={note} onClick={() => onPlay(note)} className={`teacher-key-group ${isActive ? 'active' : ''}`}>
                <rect x={x} y="0" width={whiteWidth * 0.62} height="118" rx="6" className="teacher-black-key" />
                <text x={x + whiteWidth * 0.31} y="102" textAnchor="middle" className="teacher-black-label">{note}</text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

const WIND_FINGERINGS = [
  [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
];

function WindTeacher({ instrument, activeNotes, onPlay }) {
  const note = [...activeNotes][0] || null;
  const midi = note ? safeMidi(note) : null;
  const fingering = WIND_FINGERINGS[Math.abs(Number(midi || 60)) % WIND_FINGERINGS.length];
  const trumpet = instrument === 'trumpet';
  const valvePatterns = [[0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0]];
  const valves = valvePatterns[Math.abs(Number(midi || 60)) % valvePatterns.length];
  const practiceNotes = instrument === 'clarinet'
    ? ['D3', 'E3', 'F3', 'G3', 'A3', 'B3', 'C4', 'D4']
    : ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'Bb4', 'C5'];
  return (
    <div className="instrument-teacher-surface wind-teacher">
      <div className="teacher-callout" aria-live="polite">
        <span className="teacher-live-dot" />
        <strong>{trumpet ? 'Press the glowing valves' : 'Cover the glowing keys'}</strong>
        <span>{note || 'Play a song or tap a practice note to see its fingering.'}</span>
      </div>
      <svg viewBox="0 0 1000 420" role="img" aria-label={`Interactive ${instrument} fingering view`}>
        {trumpet ? (
          <>
            <path d="M110 245 H720 Q850 245 850 160 Q850 85 760 85 H710" className="teacher-wind-tube" />
            <path d="M830 120 L950 55 V265 L830 205Z" className="teacher-brass-bell" />
            {[0, 1, 2].map((index) => (
              <g key={index} className={`teacher-wind-key ${valves[index] ? 'active' : ''}`}>
                <rect x={365 + index * 105} y="110" width="58" height="150" rx="18" />
                <circle cx={394 + index * 105} cy="95" r="34" />
                <text x={394 + index * 105} y="185" textAnchor="middle">{index + 1}</text>
              </g>
            ))}
          </>
        ) : (
          <>
            <path d={instrument === 'saxophone' ? 'M180 70 Q370 30 420 165 V320 Q420 370 520 350 L805 260' : 'M115 210 H890'} className="teacher-wind-tube" />
            {fingering.map((pressed, index) => (
              <g key={index} className={`teacher-wind-key ${pressed ? 'active' : ''}`}>
                <circle cx={245 + index * 82} cy={210} r="29" />
                <text x={245 + index * 82} y="217" textAnchor="middle">{index + 1}</text>
              </g>
            ))}
          </>
        )}
        <g className="wind-practice-notes">
          {practiceNotes.map((practiceNote, index) => (
            <g key={practiceNote} role="button" tabIndex="0" onClick={() => onPlay(practiceNote)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onPlay(practiceNote); }}>
              <rect x={110 + index * 98} y="330" width="82" height="48" rx="13" />
              <text x={151 + index * 98} y="360" textAnchor="middle">{practiceNote}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

export default function InstrumentTeacherSurface({ instrument, activeNotes, onPlay }) {
  if (instrument === 'piano') return <SynthTeacher activeNotes={activeNotes} onPlay={onPlay} />;
  if (instrument === 'upright-bass') return <VerticalBassTeacher activeNotes={activeNotes} onPlay={onPlay} />;
  if (instrument === 'drums') return <DrumTeacher activeNotes={activeNotes} onPlay={onPlay} />;
  if (instrument === 'synth') return <SynthTeacher activeNotes={activeNotes} onPlay={onPlay} />;
  if (['flute', 'saxophone', 'trumpet', 'clarinet'].includes(instrument)) return <WindTeacher instrument={instrument} activeNotes={activeNotes} onPlay={onPlay} />;
  return <FrettedInstrument instrument={instrument} activeNotes={activeNotes} onPlay={onPlay} />;
}
