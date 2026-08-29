import { COUNTRY_CALLING_CODES, countryByIso } from '../data/countryCallingCodes.js';
import { phonePreview } from '../utils/phoneNumbers.js';
import '../countryPhoneField.css';

export default function CountryPhoneField({
  countryIso,
  nationalNumber,
  onCountryChange,
  onNumberChange,
  disabled = false,
  label = 'Phone number',
  idPrefix = 'phone',
}) {
  const selected = countryByIso(countryIso);
  return (
    <fieldset className="country-phone-field" disabled={disabled}>
      <legend>{label}</legend>
      <div className="country-phone-controls">
        <label htmlFor={`${idPrefix}-country`}>
          <span>Country</span>
          <select
            id={`${idPrefix}-country`}
            value={selected.iso}
            autoComplete="tel-country-code"
            onChange={(event) => onCountryChange(event.target.value)}
          >
            {COUNTRY_CALLING_CODES.map((country) => (
              <option key={country.iso} value={country.iso}>
                {country.flag} {country.name} ({country.dialCode})
              </option>
            ))}
          </select>
        </label>
        <label htmlFor={`${idPrefix}-national`}>
          <span>Number</span>
          <div className="national-phone-input">
            <b aria-hidden="true">{selected.dialCode}</b>
            <input
              id={`${idPrefix}-national`}
              type="tel"
              inputMode="tel"
              autoComplete="tel-national"
              placeholder={selected.iso === 'SG' ? '8123 4567' : 'National number'}
              value={nationalNumber}
              onChange={(event) => onNumberChange(event.target.value.slice(0, 30))}
              required
            />
          </div>
        </label>
      </div>
      <small>
        {selected.name} {selected.dialCode}. Enter the number without {selected.dialCode}.
        {nationalNumber && <> Saved as <strong>{phonePreview(selected.iso, nationalNumber)}</strong>.</>}
      </small>
    </fieldset>
  );
}
