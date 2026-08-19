import {
  GRAND_END_MIDI,
  GRAND_START_MIDI,
} from './grandPianoLayout.js';

import {
  midiToNote,
  parseNote,
} from './noteMath.js';

const SAME_NOTE_RETRIGGER_GAP = 0.035;
const SIMULTANEOUS_WINDOW = 0.018;
const MIN_HUMAN_HOLD = 0.12;
const MAX_NOTE_SECONDS = 16;

function clamp(value, min, max) {
  return Math.max(
    min,
    Math.min(max, value)
  );
}

function quantizeTimeKey(
  time,
  window = SIMULTANEOUS_WINDOW
) {
  return String(
    Math.round(time / window)
  );
}

function makeHumanHold(rawDuration) {
  const safeDuration =
    Number.isFinite(rawDuration)
      ? clamp(
          rawDuration,
          0.055,
          MAX_NOTE_SECONDS
        )
      : 0.45;

  if (safeDuration >= 1.0) {
    return clamp(
      safeDuration * 1.03,
      MIN_HUMAN_HOLD,
      MAX_NOTE_SECONDS
    );
  }

  if (safeDuration >= 0.45) {
    return clamp(
      safeDuration * 1.08,
      MIN_HUMAN_HOLD,
      MAX_NOTE_SECONDS
    );
  }

  return clamp(
    Math.max(
      safeDuration * 1.12,
      0.16
    ),
    MIN_HUMAN_HOLD,
    MAX_NOTE_SECONDS
  );
}

function readSeconds(
  value,
  fallback = null
) {
  const number = Number(value);

  return Number.isFinite(number)
    ? number
    : fallback;
}

/*
 * Converts different pedal formats into true or false.
 *
 * Supported:
 * true / false
 * 0 / 1
 * MIDI CC64 values from 0 to 127
 * "down" / "up"
 * "on" / "off"
 * "pressed" / "released"
 */
function normalizePedalValue(value) {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return null;
    }

    /*
     * Values from 0 to 1 are normalized pedal values.
     * Values above 1 are treated as MIDI 0–127 values.
     */
    return value <= 1
      ? value >= 0.5
      : value >= 64;
  }

  const text = String(value ?? '')
    .trim()
    .toLowerCase();

  if (
    [
      'down',
      'on',
      'true',
      '1',
      'pressed',
      'sustain',
    ].includes(text)
  ) {
    return true;
  }

  if (
    [
      'up',
      'off',
      'false',
      '0',
      'released',
    ].includes(text)
  ) {
    return false;
  }

  /*
   * Handles numeric strings such as:
   * "127", "64", "1", "0"
   */
  const numericValue = Number(text);

  if (Number.isFinite(numericValue)) {
    return numericValue <= 1
      ? numericValue >= 0.5
      : numericValue >= 64;
  }

  return null;
}

/*
 * Reads pedal events from several possible JSON formats
 * and converts all of them into one standard format:
 *
 * {
 *   id: string,
 *   time: number,
 *   down: boolean,
 *   value: 127 or 0,
 *   controller: 64,
 *   source: string
 * }
 */
function normalizePedalEvents(song) {
  const candidates = [];

  const append = (
    events,
    source
  ) => {
    if (!Array.isArray(events)) {
      return;
    }

    events.forEach(
      (event, index) => {
        candidates.push({
          event,
          index,
          source,
        });
      }
    );
  };

  /*
   * Friendly JSON pedal formats.
   */
  append(
    song?.pedals,
    'pedals'
  );

  append(
    song?.pedalEvents,
    'pedalEvents'
  );

  append(
    song?.sustainPedal,
    'sustainPedal'
  );

  append(
    song?.sustain,
    'sustain'
  );

  /*
   * MIDI sustain pedal events.
   *
   * Controller 64 is the MIDI sustain pedal.
   */
  if (
    Array.isArray(
      song?.controlChanges
    )
  ) {
    song.controlChanges.forEach(
      (event, index) => {
        const controller = Number(
          event?.controller ??
          event?.control ??
          event?.cc
        );

        if (controller === 64) {
          candidates.push({
            event,
            index,
            source:
              'controlChanges',
          });
        }
      }
    );
  }

  const normalized = candidates
    .map(
      ({
        event,
        index,
        source,
      }) => {
        const time = Number(
          event?.time ??
          event?.at ??
          event?.start ??
          event?.seconds
        );

        const rawState =
          event?.down ??
          event?.isDown ??
          event?.state ??
          event?.value ??
          event?.pedal;

        const down =
          normalizePedalValue(
            rawState
          );

        if (
          !Number.isFinite(time) ||
          time < 0 ||
          down === null
        ) {
          return null;
        }

        return {
          id:
            event?.id ||
            `pedal-${source}-${index}-${time.toFixed(4)}`,

          time: Number(
            time.toFixed(4)
          ),

          down,

          /*
           * Keep a MIDI-compatible pedal value.
           */
          value: down ? 127 : 0,

          controller: 64,
          source,
        };
      }
    )
    .filter(Boolean)
    .sort((a, b) => {
      if (a.time !== b.time) {
        return a.time - b.time;
      }

      /*
       * When two events occur at exactly the same time,
       * pedal-up is processed before pedal-down.
       */
      return (
        Number(a.down) -
        Number(b.down)
      );
    });

  /*
   * Remove duplicated states.
   *
   * Example:
   * down, down, down
   *
   * becomes:
   * down
   */
  const compact = [];

  for (const event of normalized) {
    const previous =
      compact[
        compact.length - 1
      ];

    /*
     * If two events have the exact same time,
     * keep the final event at that time.
     */
    if (
      previous &&
      Math.abs(
        previous.time -
        event.time
      ) < 0.0001
    ) {
      compact[
        compact.length - 1
      ] = event;

      continue;
    }

    /*
     * Do not keep repeated identical states.
     */
    if (
      !previous ||
      previous.down !==
        event.down
    ) {
      compact.push(event);
    }
  }

  /*
   * Prevent the pedal from remaining stuck down
   * after autoplay finishes.
   *
   * If the final pedal event is down, automatically
   * add a pedal-up event shortly after the last note.
   */
  const finalPedalEvent =
    compact[
      compact.length - 1
    ];

  if (
    finalPedalEvent?.down === true &&
    Array.isArray(song?.notes) &&
    song.notes.length > 0
  ) {
    const noteEndTimes =
      song.notes
        .map((note) => {
          const time = Number(
            note?.time
          );

          const duration = Number(
            note?.audioDuration ??
            note?.duration ??
            note?.visualDuration ??
            0.4
          );

          if (
            !Number.isFinite(time)
          ) {
            return null;
          }

          return (
            time +
            (
              Number.isFinite(
                duration
              )
                ? duration
                : 0.4
            )
          );
        })
        .filter(
          Number.isFinite
        );

    if (noteEndTimes.length) {
      const lastNoteEnd =
        Math.max(
          ...noteEndTimes
        );

      compact.push({
        id:
          'pedal-auto-release-at-song-end',

        time: Number(
          (
            lastNoteEnd +
            0.1
          ).toFixed(4)
        ),

        down: false,
        value: 0,
        controller: 64,

        source:
          'automatic-song-end',
      });
    }
  }

  return compact;
}

/*
 * Finds whether the pedal should currently be
 * down or up at a particular song time.
 *
 * This is useful when resuming playback after pause.
 */
export function getPedalStateAt(
  pedals,
  time
) {
  let down = false;

  const targetTime =
    Number(time);

  if (
    !Number.isFinite(
      targetTime
    )
  ) {
    return false;
  }

  for (
    const event of pedals || []
  ) {
    if (
      event.time >
      targetTime
    ) {
      break;
    }

    down =
      Boolean(event.down);
  }

  return down;
}

function addLaneOffsets(notes) {
  const groups = new Map();

  notes.forEach((note) => {
    const key =
      `${quantizeTimeKey(
        note.time
      )}:${note.midi}`;

    const group =
      groups.get(key) || [];

    group.push(note);

    groups.set(
      key,
      group
    );
  });

  groups.forEach((group) => {
    if (group.length <= 1) {
      group[0]
        .laneOffsetPercent = 0;

      return;
    }

    const midpoint =
      (group.length - 1) / 2;

    group.forEach(
      (note, index) => {
        note.laneOffsetPercent =
          (
            index -
            midpoint
          ) * 0.34;

        note.widthScale =
          0.82;
      }
    );
  });
}

function removeImpossibleSamePitchOverlaps(
  notes,
  gapSeconds =
    SAME_NOTE_RETRIGGER_GAP
) {
  const lastByMidi =
    new Map();

  for (const note of notes) {
    const previous =
      lastByMidi.get(
        note.midi
      );

    if (previous) {
      const previousEnd =
        previous.time +
        previous.duration;

      const latestHumanRelease =
        note.time -
        gapSeconds;

      if (
        previousEnd >
        latestHumanRelease
      ) {
        const trimmed =
          Math.max(
            0.055,

            latestHumanRelease -
            previous.time
          );

        previous.duration =
          Number(
            trimmed.toFixed(4)
          );

        previous.audioDuration =
          Number(
            Math.min(
              previous.audioDuration ??
              previous.duration,

              trimmed
            ).toFixed(4)
          );

        if (
          previous
            .preserveVisualDuration !==
          true
        ) {
          previous.visualDuration =
            Number(
              Math.min(
                previous.visualDuration ??
                previous.duration,

                trimmed
              ).toFixed(4)
            );
        }

        previous.wasTrimmedForRetrigger =
          true;
      }
    }

    lastByMidi.set(
      note.midi,
      note
    );
  }
}

export function getSongDuration(
  song
) {
  if (
    !song?.notes?.length
  ) {
    return 0;
  }

  const endTimes =
    song.notes.map((note) => {
      const time =
        Number(note.time);

      const duration =
        Number(
          note.audioDuration ??
          note.duration ??
          note.visualDuration ??
          0.4
        );

      if (
        !Number.isFinite(time)
      ) {
        return 0;
      }

      return (
        time +
        (
          Number.isFinite(
            duration
          )
            ? duration
            : 0.4
        )
      );
    });

  return (
    Math.max(...endTimes) +
    0.65
  );
}

export function findStartIndex(
  notes,
  time
) {
  let low = 0;
  let high =
    notes.length;

  while (low < high) {
    const mid =
      Math.floor(
        (
          low +
          high
        ) / 2
      );

    if (
      notes[mid].time <
      time
    ) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

export function getActivePerformanceEvents(
  song,
  currentTime,
  limit = 12
) {
  if (
    !song?.notes?.length
  ) {
    return [];
  }

  const active = [];
  const lookBehind = 0.05;

  for (
    const event of song.notes
  ) {
    if (
      event.time >
      currentTime + 0.02
    ) {
      break;
    }

    const end =
      event.time +
      (
        event.visualDuration ??
        event.duration ??
        event.audioDuration ??
        0.2
      ) +
      0.03;

    if (
      end >=
      currentTime -
      lookBehind
    ) {
      active.push(event);
    }
  }

  return active
    .sort(
      (a, b) =>
        b.time -
        a.time
    )
    .slice(0, limit);
}

export function normalizeSong(
  song
) {
  const hasExactDurationFields =
    Array.isArray(song?.notes) &&
    song.notes.some((note) => (
      note?.audioDuration !== undefined ||
      note?.visualDuration !== undefined ||
      note?.scoreDuration !== undefined
    ));

  const preserveScoreDurations =
    song?.performance
      ?.preserveScoreDurations !==
      false &&
    (
      song?.performance
        ?.preserveScoreDurations ===
        true ||
      song?.performance
        ?.preserveVideoVisualDurations ===
        true ||
      song?.performance
        ?.videoOnly ===
        true ||
      song?.importedFromMidi ===
        true ||
      Boolean(song?.schemaVersion) ||
      Boolean(
        song?.performance
          ?.durationFieldPolicy
      ) ||
      hasExactDurationFields
    );

  const parsed = [
    ...(song.notes || []),
  ]
    .map(
      (note, index) => {
        const time =
          Number(note.time);

        const rawDuration =
          Number(
            note.duration ??
            0.45
          );

        const rawVisualDuration =
          readSeconds(
            note.visualDuration,
            rawDuration
          );

        const rawAudioDuration =
          readSeconds(
            note.audioDuration,
            rawDuration
          );

        const rawVelocity =
          Number(
            note.velocity ??
            0.82
          );

        let parsedNote;

        try {
          parsedNote =
            parseNote(
              note.note
            );
        } catch {
          return null;
        }

        if (
          !Number.isFinite(time) ||
          time < 0
        ) {
          return null;
        }

        /*
         * No octave folding.
         * No collision pushing.
         * No fake range remapping.
         *
         * If the JSON contains a note outside the
         * real 88-key piano range, drop it instead
         * of converting it into the wrong note.
         */
        if (
          parsedNote.midi <
            GRAND_START_MIDI ||
          parsedNote.midi >
            GRAND_END_MIDI
        ) {
          return null;
        }

        const midi =
          parsedNote.midi;

        const normalizedNote =
          midiToNote(midi);

        const duration =
          preserveScoreDurations
            ? clamp(
                rawDuration,
                0.055,
                MAX_NOTE_SECONDS
              )
            : makeHumanHold(
                rawDuration
              );

        const visualDuration =
          preserveScoreDurations
            ? clamp(
                rawVisualDuration,
                0.055,
                MAX_NOTE_SECONDS
              )
            : duration;

        const audioDuration =
          preserveScoreDurations
            ? clamp(
                rawAudioDuration,
                0.055,
                MAX_NOTE_SECONDS
              )
            : duration;

        return {
          id:
            note.id ||
            `note-${index}-${normalizedNote}-${time.toFixed(4)}`,

          note:
            normalizedNote,

          midi,

          time: Number(
            time.toFixed(4)
          ),

          duration: Number(
            duration.toFixed(4)
          ),

          visualDuration:
            Number(
              visualDuration
                .toFixed(4)
            ),

          audioDuration:
            Number(
              audioDuration
                .toFixed(4)
            ),

          originalDuration:
            Number.isFinite(
              rawDuration
            )
              ? rawDuration
              : 0.45,

          scoreDuration:
            readSeconds(
              note.scoreDuration,
              rawDuration
            ),

          velocity:
            Number.isFinite(
              rawVelocity
            )
              ? clamp(
                  rawVelocity,
                  0.04,
                  1
                )
              : 0.82,

          releaseSeconds:
            readSeconds(
              note.releaseSeconds,

              song?.performance
                ?.defaultAutoplayReleaseSeconds ??
                null
            ),

          scoreRole:
            note.scoreRole,

          originalNote:
            note.original_note ||
            note.originalNote ||
            note.note,

          measure:
            note.measure,

          source:
            note.source,

          dynamic:
            note.dynamic,

          hand:
            note.hand,

          channel:
            note.channel,

          program:
            note.program,

          rawVelocity:
            note.rawVelocity,

          softPedal:
            note.softPedal ===
            true,

          originalTick:
            note.originalTick,

          endTick:
            note.endTick,

          wasTrimmedForRetrigger:
            note
              .wasTrimmedForRetrigger ===
            true,

          preserveVisualDuration:
            preserveScoreDurations,

          laneOffsetPercent: 0,
          widthScale: 1,
        };
      }
    )
    .filter(Boolean)
    .sort((a, b) => {
      if (
        a.time !== b.time
      ) {
        return (
          a.time -
          b.time
        );
      }

      if (
        a.midi !== b.midi
      ) {
        return (
          a.midi -
          b.midi
        );
      }

      return a.id.localeCompare(
        b.id
      );
    });

  removeImpossibleSamePitchOverlaps(
    parsed,

    readSeconds(
      song?.performance
        ?.sameKeyRetriggerGapSeconds,

      SAME_NOTE_RETRIGGER_GAP
    )
  );

  addLaneOffsets(parsed);

  const pedals =
    normalizePedalEvents(song);

  return {
    title:
      song.title ||
      'Untitled Song',

    composer:
      song.composer ||
      song.artist ||
      song.arrangement ||
      'Unknown',

    bpm:
      Number.isFinite(
        Number(
          song.bpm ??
          song.tempo
        )
      )
        ? Number(
            song.bpm ??
            song.tempo
          )
        : 90,

    tempo:
      Number.isFinite(
        Number(
          song.tempo ??
          song.bpm
        )
      )
        ? Number(
            song.tempo ??
            song.bpm
          )
        : undefined,

    timeSignature:
      song.timeSignature ||
      '4/4',

    key:
      song.key,

    performance:
      song.performance,

    schemaVersion:
      song.schemaVersion,

    instrument:
      song.instrument ||
      'piano',

    playbackMode:
      song.playbackMode ||
      'instrumental',

    vocalMelodyIncluded:
      song.vocalMelodyIncluded ===
      true,

    artist:
      song.artist,

    slug:
      song.slug,

    source:
      song.source,

    tempoEvents:
      song.tempoEvents,

    importedFromMidi:
      song.importedFromMidi ===
      true,

    normalizedFormat:
      song.normalizedFormat,

    midi:
      song.midi,

    controlChanges:
      song.controlChanges,

    midiProgramChanges:
      song.midiProgramChanges,

    percussionEvents:
      song.percussionEvents,

    pedals,
    notes: parsed,
  };
}
