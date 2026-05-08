from __future__ import annotations

import json
import re
from typing import Any

import requests
from bs4 import BeautifulSoup
import pandas as pd


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dicts(item)


def _extract_from_json_payload(payload: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in _iter_dicts(payload):
        # area
        for k in ("area", "floorSize", "size", "m2"):
            if k in d and out.get("m") is None:
                m = _to_float(d.get(k))
                if m is not None and 5 <= m <= 10000:
                    out["m"] = m

        # rooms
        for k in ("roomsNumber", "numberOfRooms", "rooms", "noRooms"):
            if k in d and out.get("no_rooms") is None:
                rooms = _to_float(d.get(k))
                if rooms is not None and 0 <= rooms <= 50:
                    out["no_rooms"] = rooms

        # geo
        lat = d.get("latitude", d.get("lat"))
        lon = d.get("longitude", d.get("lng", d.get("lon")))
        lat_f, lon_f = _to_float(lat), _to_float(lon)
        if lat_f is not None and lon_f is not None:
            out["map_lat"] = lat_f
            out["map_lon"] = lon_f
            out["district_lat"] = lat_f
            out["district_lon"] = lon_f

        # market
        for k in ("marketType", "market", "rynek"):
            if k in d and out.get("market") is None:
                val = str(d.get(k, "")).lower()
                if any(x in val for x in ("wtór", "wtor", "secondary")):
                    out["market"] = "secondary"
                elif any(x in val for x in ("pierwot", "primary")):
                    out["market"] = "primary"

        # publish / creation date (best-effort)
        for k in (
            "datePublished",
            "createdAt",
            "created_at",
            "publicationDate",
            "publishedAt",
            "dateCreated",
        ):
            if k in d and out.get("__created_at__") is None:
                ts = pd.to_datetime(d.get(k), errors="coerce")
                if pd.notna(ts):
                    out["__created_at__"] = str(ts)

    return out


def extract_otodom_features(url: str, timeout_sec: int = 15) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, timeout=timeout_sec, headers=headers)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    out: dict[str, Any] = {}

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title = og_title["content"].strip() or title

    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()
    og_desc = soup.find("meta", attrs={"property": "og:description"})
    if og_desc and og_desc.get("content"):
        description = og_desc["content"].strip() or description

    scripts = soup.find_all("script")
    for s in scripts:
        txt = s.string or s.get_text(strip=True)
        if not txt:
            continue
        txt = txt.strip()
        if not txt:
            continue
        if s.get("type") == "application/ld+json":
            try:
                data = json.loads(txt)
            except json.JSONDecodeError:
                continue
            out.update({k: v for k, v in _extract_from_json_payload(data).items() if k not in out})

    # Try to parse __NEXT_DATA__ payload if available.
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and (next_data.string or "").strip():
        try:
            data = json.loads(next_data.string)
            out.update({k: v for k, v in _extract_from_json_payload(data).items() if k not in out})
        except json.JSONDecodeError:
            pass

    # Text regex fallback
    page_text = soup.get_text(" ", strip=True)
    if out.get("m") is None:
        m_match = re.search(r"(\d+(?:[.,]\d+)?)\s*m(?:2|²)", page_text, flags=re.I)
        if m_match:
            out["m"] = _to_float(m_match.group(1))
    if out.get("no_rooms") is None:
        r_match = re.search(r"(\d+(?:[.,]\d+)?)\s*pok", page_text, flags=re.I)
        if r_match:
            out["no_rooms"] = _to_float(r_match.group(1))

    text = (title + " " + description).strip()
    if title:
        out["__title__"] = title
    if description:
        out["__description__"] = description
    if text:
        out["__text__"] = text

    # Safe defaults for model compatibility.
    out.setdefault("market", "secondary")
    return out

