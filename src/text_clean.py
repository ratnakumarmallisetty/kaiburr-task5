"""
Light, safe text cleaning:
- strip
- collapse whitespace
- remove URLs
We keep numbers/punctuation because they can be useful.
"""
from __future__ import annotations
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("\n", " ").replace("\r", " ")
    s = URL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s
