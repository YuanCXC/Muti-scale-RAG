from __future__ import annotations

import html
import math
import re
import unicodedata

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
PARENTHETICAL_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_title(text: str) -> str:
    value = html.unescape(str(text or "")).replace("_", " ")
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return normalize_space(value)


def title_aliases(title: str) -> set[str]:
    aliases = {normalize_title(title)}
    base = PARENTHETICAL_SUFFIX.sub("", html.unescape(title)).strip()
    if base:
        aliases.add(normalize_title(base))
    return {alias for alias in aliases if alias}


def terms(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(str(text or ""))}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def anchor_score(hit_sent_id: int, bridge_sent_id: int, sigma: float) -> float:
    return math.exp(-abs(hit_sent_id - bridge_sent_id) / max(sigma, 1e-6))
