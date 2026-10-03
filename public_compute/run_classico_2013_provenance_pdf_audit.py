from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "classico_2013_provenance_pdf_audit_config.json").read_text())


def download(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 WFL-public-document-audit/1.0",
            "Accept": "application/pdf,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def extract_text(blob: bytes, tmp: Path) -> tuple[str, int]:
    tmp.write_bytes(blob)
    reader = PdfReader(str(tmp))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts), len(reader.pages)


def contexts(text: str, keyword: str, width: int) -> list[str]:
    out = []
    low = text.casefold()
    target = keyword.casefold()
    start = 0
    while True:
        i = low.find(target, start)
        if i < 0:
            break
        lo = max(0, i - width)
        hi = min(len(text), i + len(keyword) + width)
        snippet = re.sub(r"\s+", " ", text[lo:hi]).strip()
        out.append(snippet)
        start = i + max(1, len(target))
        if len(out) >= 20:
            break
    return out


def infer_flags(doc_id: str, text: str) -> dict:
    low = re.sub(r"\s+", " ", text.casefold())
    return {
        "mentions_adamss": "adamss" in low,
        "mentions_unimi": "università degli studi di milano" in low or "universita degli studi di milano" in low,
        "mentions_2011_cert_dates": ("4 dicembre 2011" in low) or ("15 dicembre 2011" in low),
        "mentions_certification": "certific" in low,
        "mentions_extraction_system": "sistema estrazionale" in low,
        "mentions_device": "dispositivo" in low,
        "mentions_generator": "generatore" in low,
        "mentions_algorithm": "algoritmo" in low,
        "mentions_collaudo": "collaudo" in low,
        "mentions_software": "software" in low,
        "mentions_hardware": "hardware" in low,
        "mentions_2012_3925": "2012/3925" in low or "3925/giochi" in low,
        "classico_specific": doc_id in {"2013_505","2013_1523","2013_2381"},
    }


def main() -> None:
    out = {
        "schema": "wfl-classico-2013-provenance-pdf-audit-result-1",
        "config": CFG,
        "documents": [],
        "decision": {
            "classico_2013_uses_2011_adamss_certified_device": "UNKNOWN",
            "new_classico_specific_2013_device_or_certificate": "UNKNOWN",
        },
        "interpretation": "Public-document text audit only. Silence is not evidence of reuse or replacement."
    }
    tmp_dir = ROOT / ".tmp_classico_pdf_audit"
    tmp_dir.mkdir(exist_ok=True)
    width = int(CFG["context_chars"])

    for doc in CFG["documents"]:
        blob = download(doc["url"])
        sha = hashlib.sha256(blob).hexdigest()
        tmp = tmp_dir / f"{doc['id']}.pdf"
        text, pages = extract_text(blob, tmp)
        kws = {}
        for kw in CFG["keywords"]:
            hits = contexts(text, kw, width)
            if hits:
                kws[kw] = hits
        flags = infer_flags(doc["id"], text)
        out["documents"].append({
            "id": doc["id"],
            "protocol": doc["protocol"],
            "source_url": doc["url"],
            "sha256": sha,
            "bytes": len(blob),
            "pages": pages,
            "extracted_text_chars": len(text),
            "flags": flags,
            "keyword_contexts": kws,
        })

    classico = [d for d in out["documents"] if d["flags"]["classico_specific"]]
    explicit_old = [
        d for d in classico
        if d["flags"]["mentions_adamss"]
        and (d["flags"]["mentions_2011_cert_dates"] or d["flags"]["mentions_2012_3925"])
    ]
    explicit_new = [
        d for d in classico
        if d["flags"]["mentions_certification"]
        and ("ADAMSS" in d["keyword_contexts"] or "collaudo" in d["keyword_contexts"])
    ]
    if explicit_old:
        out["decision"]["classico_2013_uses_2011_adamss_certified_device"] = "POSSIBLE_EXPLICIT_REFERENCE_REQUIRES_HUMAN_CONTEXT_REVIEW"
    if explicit_new:
        out["decision"]["new_classico_specific_2013_device_or_certificate"] = "POSSIBLE_EXPLICIT_REFERENCE_REQUIRES_HUMAN_CONTEXT_REVIEW"

    out_path = ROOT / "results" / "classico_2013_provenance_pdf_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
