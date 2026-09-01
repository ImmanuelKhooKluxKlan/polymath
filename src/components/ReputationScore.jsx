function points(value) {
  const number = Number(value) || 0;
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

export default function ReputationScore({ ranking, audienceLabel = 'buyers', compact = false }) {
  const ratingPoints = Number(ranking?.ratingPoints) || 0;
  const audiencePoints = Number(ranking?.audiencePoints) || 0;
  const totalPoints = Number(ranking?.totalPoints) || 0;
  const label = `${points(totalPoints)} out of 50 ranking points: ${points(ratingPoints)} from ratings and ${points(audiencePoints)} from ${audienceLabel}`;

  return (
    <div className={`reputation-score ${compact ? 'compact' : ''}`} aria-label={label}>
      <span className="reputation-total"><strong>{points(totalPoints)}</strong><small>/50</small></span>
      <span className="reputation-part"><small>Rating</small><b>{points(ratingPoints)}/10</b></span>
      <span className="reputation-part"><small>{audienceLabel}</small><b>{points(audiencePoints)}/40</b></span>
    </div>
  );
}
