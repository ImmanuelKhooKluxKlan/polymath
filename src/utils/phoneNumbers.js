import { countryByIso } from '../data/countryCallingCodes.js';

const SIGNIFICANT_LEADING_ZERO_COUNTRIES = new Set(['IT', 'SM', 'VA']);

export function internationalPhone(countryIso, nationalValue) {
  const raw = String(nationalValue || '').trim();
  if (raw.startsWith('+')) {
    const pastedDigits = raw.replace(/\D/g, '');
    return /^\d{8,15}$/.test(pastedDigits) && pastedDigits[0] !== '0'
      ? `+${pastedDigits}`
      : '';
  }

  const country = countryByIso(countryIso);
  let nationalDigits = raw.replace(/\D/g, '');
  if (!SIGNIFICANT_LEADING_ZERO_COUNTRIES.has(country.iso)) {
    nationalDigits = nationalDigits.replace(/^0+/, '');
  }
  const combined = `${country.dialDigits}${nationalDigits}`;
  return /^[1-9]\d{7,14}$/.test(combined) ? `+${combined}` : '';
}

export function phonePreview(countryIso, nationalValue) {
  const country = countryByIso(countryIso);
  const phone = internationalPhone(countryIso, nationalValue);
  return phone || `${country.dialCode} ${String(nationalValue || '').trim()}`.trim();
}
