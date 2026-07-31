const GLYPHS = {
  piano: 'PI',
  guitar: 'GT',
  fiddle: 'FI',
  banjo: 'BJ',
  mandolin: 'MN',
  dobro: 'DB',
  'upright-bass': 'BS',
  ukulele: 'UK',
  'electric-guitar': 'EG',
  drums: 'DR',
  synth: 'SY',
  violin: 'VN',
  cello: 'VC',
  flute: 'FL',
  saxophone: 'SX',
  trumpet: 'TR',
  clarinet: 'CL',
};

export default function InstrumentIcon({ instrument, size = 'md' }) {
  return (
    <span className={`instrument-icon instrument-icon-${size}`} aria-hidden="true">
      <svg viewBox="0 0 44 44" focusable="false">
        <circle cx="22" cy="22" r="19" />
        <path d="M13 29c5-2 7-7 8-15m10 1c-6 2-8 7-9 15M16 12l15 20" />
        <text x="22" y="26" textAnchor="middle">{GLYPHS[instrument] || 'MU'}</text>
      </svg>
    </span>
  );
}
