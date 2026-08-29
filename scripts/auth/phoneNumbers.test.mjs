import assert from 'node:assert/strict';
import test from 'node:test';

import { COUNTRY_CALLING_CODES, countryByIso, suggestedCountryIso } from '../../src/data/countryCallingCodes.js';
import { internationalPhone } from '../../src/utils/phoneNumbers.js';

test('combines the selected country with a national phone number', () => {
  assert.equal(internationalPhone('SG', '8123 4567'), '+6581234567');
  assert.equal(internationalPhone('MY', '012-345 6789'), '+60123456789');
  assert.equal(internationalPhone('US', '(415) 555-0123'), '+14155550123');
});

test('keeps significant Italian leading zeroes and accepts pasted international numbers', () => {
  assert.equal(internationalPhone('IT', '06 698 12345'), '+390669812345');
  assert.equal(internationalPhone('SG', '+44 7700 900123'), '+447700900123');
});

test('rejects incomplete numbers and falls back safely to Singapore', () => {
  assert.equal(internationalPhone('SG', '123'), '');
  assert.equal(countryByIso('invalid').iso, 'SG');
  assert.equal(suggestedCountryIso('en-US'), 'US');
  assert.equal(suggestedCountryIso('en'), 'SG');
});

test('country selector covers an alphabetized, unique A-to-Z collection', () => {
  assert.ok(COUNTRY_CALLING_CODES.length >= 240);
  assert.equal(new Set(COUNTRY_CALLING_CODES.map((country) => country.iso)).size, COUNTRY_CALLING_CODES.length);
  const names = COUNTRY_CALLING_CODES.map((country) => country.name);
  assert.deepEqual(names, [...names].sort((left, right) => left.localeCompare(right, 'en')));
  assert.ok(COUNTRY_CALLING_CODES.some((country) => country.name.startsWith('A')));
  assert.ok(COUNTRY_CALLING_CODES.some((country) => country.name.startsWith('Z')));
});
