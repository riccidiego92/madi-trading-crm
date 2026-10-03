from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "classico_2013_provenance_pdf_audit_config.json").read_text())
DOC_IDS = {"2013_505", "2013_1523", "2013_2381"}
KEYWORDS = CFG["keywords"]
WIDTH = int(CFG["context_chars"])


def download(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 WFL-public-document-ocr-audit/1.0",
            "Accept": "application/pdf,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def contexts(text: str, keyword: str) -> list[str]:
    out = []
    low = text.casefold()
    target = keyword.casefold()
    start = 0
    while True:
        i = low.find(target, start)
        if i < 0:
            break
        lo = max(0, i - WIDTH)
        hi = min(len(text), i + len(keyword) + WIDTH)
        out.append(norm(text[lo:hi]))
        start = i + max(1, len(target))
        if len(out) >= 20:
            break
    return out


def ocr_pdf(pdf_path: Path, work: Path) -> tuple[str, list[dict]]:
    stem = work / "page"
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", "200", str(pdf_path), str(stem)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pages = sorted(work.glob("page-*.jpg"))
    texts = []
    page_meta = []
    for idx, img in enumerate(pages, start=1):
        cp = subprocess.run(
            ["tesseract", str(img), "stdout", "-l", "ita+eng", "--psm", "6"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        txt = cp.stdout or ""
        texts.append(txt)
        page_meta.append({
            "page": idx,
            "ocr_chars": len(txt),
            "sha256_text": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
        })
    return "\n\n".join(texts), page_meta


def main() -> None:
    tmp = ROOT / ".tmp_classico_2013_ocr"
    tmp.mkdir(exist_ok=True)
    result = {
        "schema": "wfl-classico-2013-provenance-ocr-audit-result-1",
        "status": "OCR_COMPLETE_CONTEXT_REVIEW_REQUIRED",
        "ocr": {"dpi": 200, "languages": "ita+eng", "psm": 6, "all_pages": True},
        "documents": [],
        "decision": {
            "classico_2013_uses_2011_adamss_certified_device": "UNKNOWN",
            "new_classico_specific_2013_device_or_certificate": "UNKNOWN"
        },
        "interpretation": "OCR of public official scans. Keyword hits are evidence leads only; explicit context is required for any provenance conclusion."
    }
    for doc in CFG["documents"]:
        if doc["id"] not in DOC_IDS:
            continue
        blob = download(doc["url"])
        pdf_path = tmp / f"{doc['id']}.pdf"
        pdf_path.write_bytes(blob)
        work = tmp / doc["id"]
        work.mkdir(exist_ok=True)
        text, page_meta = ocr_pdf(pdf_path, work)
        kws = {}
        for kw in KEYWORDS:
            hits = contexts(text, kw)
            if hits:
                kws[kw] = hits
        low = text.casefold()
        flags = {
            "mentions_adamss": "adamss" in low,
            "mentions_unimi": "università degli studi di milano" in low or "universita degli studi di milano" in low,
            "mentions_certification": "certific" in low,
            "mentions_device": "dispositivo" in low,
            "mentions_extraction_system": "sistema estrazionale" in low,
            "mentions_generator": "generatore" in low,
            "mentions_algorithm": "algoritmo" in low,
            "mentions_collaudo": "collaudo" in low,
            "mentions_2011": "2011" in low,
            "mentions_2012_3925": "2012/3925" in low or "3925/giochi" in low,
        }
        result["documents"].append({
            "id": doc["id"],
            "protocol": doc["protocol"],
            "source_url": doc["url"],
            "pdf_sha256": hashlib.sha256(blob).hexdigest(),
            "pdf_bytes": len(blob),
            "ocr_text_chars": len(text),
            "ocr_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "pages": page_meta,
            "flags": flags,
            "keyword_contexts": kws,
        })

    old_refs = [
        d for d in result["documents"]
        if d["flags"]["mentions_adamss"]
        and (d["flags"]["mentions_2011"] or d["flags"]["mentions_2012_3925"])
    ]
    if old_refs:
        result["decision"]["classico_2013_uses_2011_adamss_certified_device"] = "POSSIBLE_EXPLICIT_REFERENCE_REQUIRES_CONTEXT_REVIEW"

    new_refs = [
        d for d in result["documents"]
        if d["flags"]["mentions_certification"]
        and (d["flags"]["mentions_device"] or d["flags"]["mentions_extraction_system"])
    ]
    if new_refs:
        result["decision"]["new_classico_specific_2013_device_or_certificate"] = "POSSIBLE_REFERENCE_REQUIRES_CONTEXT_REVIEW"

    out = ROOT / "results" / "classico_2013_provenance_ocr_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
