"""Turn base/candidate behavior JSON into a concise human review document."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    base_by_id = {row["id"]: row for row in base["results"]}
    rows = []
    categories: dict[str, list[float]] = defaultdict(list)
    for row in candidate["results"]:
        delta = row["score"] - base_by_id[row["id"]]["score"]
        rows.append({**row, "delta": delta})
        categories[row["category"]].append(row["score"])

    improved = sorted((row for row in rows if row["delta"] > 0), key=lambda row: row["delta"], reverse=True)
    regressed = sorted((row for row in rows if row["delta"] < 0), key=lambda row: row["delta"])
    weak = sorted((row for row in rows if row["score"] < 0.85), key=lambda row: row["score"])
    unsafe = [row for row in rows if not row["safetyPassed"]]
    gate_lines = [
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in decision["gates"].items()
    ]
    lines = [
        f"# ChatBoss {args.version} human review",
        "",
        "> A behavior score is a fixed rubric result, not a claim of real-world accuracy.",
        "",
        "## Headline",
        "",
        f"- Frozen base: **{percent(base['overallScore'])}**",
        f"- Candidate: **{percent(candidate['overallScore'])}**",
        f"- Change: **{candidate['overallScore'] - base['overallScore']:+.4f}**",
        f"- Candidate safety pass: **{percent(candidate['safetyPassRate'])}**",
        f"- Cases: **{candidate['caseCount']}**",
        f"- Automatic promotion: **disabled**",
        "",
        "## Gates",
        "",
        *gate_lines,
        "",
        "## Role scores",
        "",
        "| Role | Base | Candidate |",
        "| --- | ---: | ---: |",
        *[
            f"| {role} | {percent(base['roleScores'][role])} | {percent(score)} |"
            for role, score in candidate["roleScores"].items()
        ],
        "",
        "## Weak cases below 85%",
        "",
        "| Case | Category | Score | Change | Safety |",
        "| --- | --- | ---: | ---: | --- |",
        *[
            f"| `{row['id']}` | {row['category']} | {percent(row['score'])} | {row['delta']:+.3f} | {'pass' if row['safetyPassed'] else 'FAIL'} |"
            for row in weak
        ],
        "",
        "## Largest improvements",
        "",
        *([f"- `{row['id']}`: {row['delta']:+.3f}" for row in improved[:10]] or ["- None"]),
        "",
        "## Regressions",
        "",
        *([f"- `{row['id']}`: {row['delta']:+.3f}" for row in regressed] or ["- None"]),
        "",
        "## Safety failures",
        "",
        *([f"- `{row['id']}`: {row['output']}" for row in unsafe] or ["- None"]),
        "",
        "## Category scores",
        "",
        "| Category | Candidate mean | Cases |",
        "| --- | ---: | ---: |",
        *[
            f"| {category} | {percent(sum(scores) / len(scores))} | {len(scores)} |"
            for category, scores in sorted(categories.items())
        ],
        "",
        "## Decision",
        "",
        "Do not deploy solely from this document. Review every weak, regressed, and safety-failing output before approving an adapter.",
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "weakCases": len(weak),
        "regressions": len(regressed),
        "safetyFailures": len(unsafe),
    }, indent=2))


if __name__ == "__main__":
    main()
