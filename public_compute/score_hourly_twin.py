from __future__ import annotations

from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

ROME = ZoneInfo("Europe/Rome")
BASE = "https://www.winforlife.it/archivio-estrazioni-classico"
MONTH_IT = {
    1: "gennaio", 2: "febbraio", 3: "marzo", 4: "aprile", 5: "maggio",
    6: "giugno", 7: "luglio", 8: "agosto", 9: "settembre",
    10: "ottobre", 11: "novembre", 12: "dicembre",
}


def _canonical_hash(payload: dict) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def contest_number(d: date, hour: int) -> int:
    launch = date(2013, 1, 21)
    if d < launch:
        raise ValueError("date predates launch")
    if d == launch:
        if hour not in range(12, 24):
            raise ValueError("invalid launch-day hour")
        return 1 + (hour - 12)
    if hour not in range(7, 24):
        raise ValueError("invalid Classico hour")
    if d.year == 2013:
        days_after = (d - launch).days
        first = 13 + (days_after - 1) * 17
        return first + (hour - 7)
    doy = d.timetuple().tm_yday
    return 17 * (doy - 1) + 1 + (hour - 7)


def detail_url(d: date, contest: int) -> str:
    return (
        f"{BASE}/concorso-{int(contest)}/"
        f"{d.day}-{MONTH_IT[d.month]}-{d.year}"
    )


def fetch_official(target: datetime) -> dict | None:
    d = target.date()
    contest = contest_number(d, target.hour)
    url = detail_url(d, contest)
    r = requests.get(
        url,
        timeout=25,
        headers={"User-Agent": "WFL-Classico-Lab/0.1 prospective scorer"},
    )
    if r.status_code != 200:
        return None
    raw = r.content
    text = BeautifulSoup(raw, "lxml").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    number = r"(?:20|1\d|[1-9])"
    m = re.search(
        rf"Combinazione\s+vincente\s+Numerone\s+"
        rf"((?:{number}\s+){{10}})Numerone\s+({number})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    main = sorted(int(x) for x in re.findall(number, m.group(1)))
    numerone = int(m.group(2))
    if len(main) != 10 or len(set(main)) != 10:
        raise ValueError("official detail returned invalid main combination")
    return {
        "date": d.isoformat(),
        "time": f"{target.hour:02d}:00",
        "contest": contest,
        "main": main,
        "numerone": numerone,
        "source_url": url,
        "source_sha256": sha256(raw).hexdigest(),
    }


def verify_freeze(freeze: dict) -> datetime:
    if freeze.get("schema") != "wfl-hourly-twin-development-freeze-1":
        raise ValueError("unexpected freeze schema")
    saved = freeze.get("freeze_sha256")
    unsigned = dict(freeze)
    unsigned.pop("freeze_sha256", None)
    if _canonical_hash(unsigned) != saved:
        raise ValueError("freeze hash mismatch")
    target = datetime.fromisoformat(freeze["target"]["datetime"]).astimezone(ROME)
    frozen = datetime.fromisoformat(freeze["frozen_at"]).astimezone(ROME)
    if frozen >= target:
        raise ValueError("freeze does not precede target")
    return target


def score_one(path: Path, score_root: Path, now: datetime) -> dict | None:
    freeze = json.loads(path.read_text())
    target = verify_freeze(freeze)
    if now.astimezone(ROME) <= target:
        return None

    rel = path.relative_to(Path("public_compute/prospective/hourly"))
    dest = score_root / rel
    if dest.exists():
        return {"status": "EXISTS_IMMUTABLE", "path": str(dest)}

    official = fetch_official(target)
    if official is None:
        return None

    pred = tuple(int(x) for x in freeze["prediction"]["main"])
    comp = tuple(int(x) for x in freeze["prediction"]["complement"])
    actual = tuple(int(x) for x in official["main"])
    k = len(set(pred) & set(actual))
    num_hit = int(freeze["prediction"]["numerone"]) == int(official["numerone"])

    payload = {
        "schema": "wfl-hourly-twin-development-score-1",
        "evidence_grade": "DEVELOPMENT_PROSPECTIVE_SCORE_NO_EDGE_CLAIM",
        "freeze_sha256": freeze["freeze_sha256"],
        "scored_at": now.astimezone(ROME).isoformat(),
        "target": freeze["target"],
        "candidate_id": freeze["machine"]["candidate_id"],
        "prediction": freeze["prediction"],
        "official": official,
        "score": {
            "direct_main_overlap": int(k),
            "symmetric_main_overlap": int(max(k, 10 - k)),
            "exact_main": bool(pred == actual),
            "exact_complement": bool(comp == actual),
            "numerone_exact": bool(num_hit),
            "exact_main_plus_numerone": bool(pred == actual and num_hit),
            "exact_complement_plus_numerone": bool(comp == actual and num_hit),
        },
        "guardrails": {
            "freeze_hash_verified": True,
            "freeze_precedes_target": True,
            "official_source_retained": True,
            "post_result_retuning": False,
            "failures_retained": True,
        },
    }
    payload["score_sha256"] = _canonical_hash(payload)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"status": "SCORED", "path": str(dest), "score": payload["score"]}


def main() -> None:
    freeze_root = Path("public_compute/prospective/hourly")
    score_root = Path("public_compute/prospective/hourly_scores")
    now = datetime.now(ROME)
    results = []
    if freeze_root.exists():
        for path in sorted(freeze_root.glob("*/*.json")):
            result = score_one(path, score_root, now)
            if result is not None:
                results.append(result)
    print(json.dumps({"processed": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
