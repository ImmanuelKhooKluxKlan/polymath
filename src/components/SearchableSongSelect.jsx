import { useMemo, useState } from 'react';

function normalize(value) {
  return String(value || '').trim().toLocaleLowerCase();
}

export default function SearchableSongSelect({
  id,
  choices,
  value,
  onChange,
  busy = false,
}) {
  const [query, setQuery] = useState('');
  const visibleChoices = useMemo(() => {
    const needle = normalize(query);
    if (!needle) return choices;
    return choices
      .filter((choice) => normalize(`${choice.label} ${choice.searchText || ''}`).includes(needle))
      .sort((left, right) => {
        const leftStarts = normalize(left.label).startsWith(needle);
        const rightStarts = normalize(right.label).startsWith(needle);
        if (leftStarts !== rightStarts) return leftStarts ? -1 : 1;
        return left.label.localeCompare(right.label);
      });
  }, [choices, query]);
  const selected = choices.find((choice) => choice.id === value);
  const options = selected && !visibleChoices.some((choice) => choice.id === selected.id)
    ? [selected, ...visibleChoices]
    : visibleChoices;

  return (
    <div className="song-library-picker">
      <label className="field" htmlFor={`${id}-search`}>
        Search songs
        <input
          id={`${id}-search`}
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Type a song or artist"
          autoComplete="off"
        />
      </label>
      <label className="field" htmlFor={id}>
        Song
        <select id={id} value={value} onChange={(event) => onChange(event.target.value)} disabled={busy}>
          {options.map((choice) => (
            <option key={choice.id} value={choice.id}>{choice.label}</option>
          ))}
        </select>
      </label>
      {query && <small className="song-search-count">{visibleChoices.length} found</small>}
    </div>
  );
}
