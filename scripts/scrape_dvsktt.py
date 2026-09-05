#!/usr/bin/env python3
"""
Scrape Đại Việt Sử Ký Toàn Thư (bản fulltext nomfoundation).

Mỗi "mục" (section) có nhiều "trang" (woodblock page), phân trang phía server:
POST multipart {curPg: n} tới URL mục -> HTML chứa 1 bảng:
  - hàng có nhiều chữ Hán  = Hán + phiên âm xen kẽ (block: <Hán> [tr*dòng*cột] <phiên âm.>)
  - ô "Dịch Quốc Ngữ"      = bản dịch tiếng Việt

Kết quả: 1 file JSONL / mục, mỗi dòng = 1 trang:
  {"section","page_idx","page_label","han_phienam","dich"}

Chạy (trong env có `requests`):
  python scrape_dvsktt.py --sections 1-Ky-Hong-Bang-thi --out data/raw
  python scrape_dvsktt.py --all --out data/raw          # toàn bộ (chậm)
"""
import argparse
import json
import os
import re
import sys
import time

try:  # môi trường WSL/conda có sẵn
    import requests
except ImportError:  # máy trần chỉ có stdlib -> dùng urllib
    requests = None
import mimetypes
import urllib.error
import urllib.request
import uuid

ROOT = "https://www.nomfoundation.org/nom-project/history-of-greater-vietnam"
INDEX_URL = f"{ROOT}/Fulltext?uiLang=vn"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; sentence-alignment coursework)"}

CJK = r"一-鿿㐀-䶿\U00020000-\U0002a6df"
MARKER_RE = re.compile(r"\[\s*\d+[ab]?\s*\*\s*\d+\s*\*\s*\d+\s*\]")


def clean(s: str) -> str:
    import html as _html
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = s.replace("\xa0", " ")
    return re.sub(r"[ \t\r\n]+", " ", s).strip()


def _urlopen(req, tries):
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError) as e:
            print(f"  retry {i+1}: {e}", file=sys.stderr)
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"request failed: {req.full_url}")


def _multipart(fields):
    """Đóng gói multipart/form-data thủ công (khớp form `search_en` của site)."""
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")
    body += f"--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def get(url, session=None, tries=5):
    if requests is not None and session is not None:
        for i in range(tries):
            try:
                r = session.get(url, headers=HEADERS, timeout=30)
                r.encoding = "utf-8"
                if r.ok:
                    return r.text
            except requests.RequestException as e:
                print(f"  GET retry {i+1}: {e}", file=sys.stderr)
            time.sleep(2 * (i + 1))
        raise RuntimeError(f"GET failed: {url}")
    return _urlopen(urllib.request.Request(url, headers=HEADERS), tries)


def post_page(url, page, session=None, tries=5):
    if requests is not None and session is not None:
        for i in range(tries):
            try:
                r = session.post(
                    url, headers=HEADERS, files={"curPg": (None, str(page))}, timeout=30
                )
                r.encoding = "utf-8"
                if r.ok:
                    return r.text
            except requests.RequestException as e:
                print(f"  POST retry {i+1}: {e}", file=sys.stderr)
            time.sleep(2 * (i + 1))
        raise RuntimeError(f"POST failed: {url} curPg={page}")
    body, ctype = _multipart({"curPg": str(page)})
    req = urllib.request.Request(
        url, data=body, headers={**HEADERS, "Content-Type": ctype}, method="POST"
    )
    return _urlopen(req, tries)


def list_sections(session):
    html = get(INDEX_URL, session)
    slugs, seen = [], set()
    for m in re.finditer(r'href="Fulltext/([^"?]+)\?uiLang=vn"', html):
        slug = m.group(1)
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    return slugs


def page_count(html):
    m = re.search(r"\[\s*(\d+)\s*trang\s*\]", html)
    return int(m.group(1)) if m else None


def parse_content(html):
    """Trả về (page_label, han_phienam, dich) hoặc None nếu không có nội dung."""
    tables = re.findall(r"<table\b[^>]*>(.*?)</table>", html, re.S | re.I)
    best = None
    for t in tables:
        if ("Dịch Quốc Ngữ" in t) or MARKER_RE.search(t) or re.search(f"[{CJK}]", t):
            best = t
            break
    if best is None:
        return None
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", best, re.S | re.I)
    han_cell, dich_cell, page_label = "", "", ""
    for row in rows:
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S | re.I)
        for c in cells:
            raw = c
            txt = clean(c)
            n_cjk = len(re.findall(f"[{CJK}]", raw))
            if "Dịch Quốc Ngữ" in raw or txt.startswith("Dịch Quốc Ngữ"):
                dich_cell = txt
            elif n_cjk > len(re.findall(f"[{CJK}]", han_cell or "")):
                han_cell = txt
            m = re.search(r"Trang\s*:\s*([0-9]+[ab]?)", txt)
            if m and not page_label:
                page_label = m.group(1)
    dich_cell = re.sub(r"^\s*Dịch Quốc Ngữ\s*", "", dich_cell).strip()
    if not han_cell and not dich_cell:
        return None
    return page_label, han_cell, dich_cell


def scrape_section(slug, session, out_dir, max_pages=None, delay=1.0):
    url = f"{ROOT}/Fulltext/{slug}?uiLang=vn"
    first = get(url, session)
    n = page_count(first) or 1
    if max_pages:
        n = min(n, max_pages)
    out_path = os.path.join(out_dir, f"{slug}.jsonl")
    got = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in range(n):
            html = first if p == 0 else post_page(url, p, session)
            parsed = parse_content(html)
            if parsed is None:
                print(f"  [{slug}] page {p}: no content", file=sys.stderr)
                continue
            label, han, dich = parsed
            rec = {
                "section": slug,
                "page_idx": p,
                "page_label": label,
                "han_phienam": han,
                "dich": dich,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            got += 1
            time.sleep(delay)
    print(f"[{slug}] wrote {got}/{n} pages -> {out_path}")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sections", nargs="*", help="danh sách slug cụ thể")
    ap.add_argument("--all", action="store_true", help="scrape tất cả mục")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê mục rồi thoát")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = requests.Session() if requests is not None else None

    if args.list or (not args.sections and not args.all):
        secs = list_sections(session)
        print(f"{len(secs)} sections:")
        for s in secs:
            print(" ", s)
        if args.list or not args.all:
            return

    sections = args.sections if args.sections else list_sections(session)
    total = 0
    for slug in sections:
        try:
            total += scrape_section(
                slug, session, args.out, max_pages=args.max_pages, delay=args.delay
            )
        except Exception as e:
            print(f"!! section {slug} failed: {e}", file=sys.stderr)
    print(f"TOTAL pages scraped: {total}")


if __name__ == "__main__":
    main()
