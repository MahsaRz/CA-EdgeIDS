"""Check generated text files across Git's LF/CRLF conversion.

This is only for generated configs and samples. Official archive checks continue
to compare the exact original bytes against the archive manifests.
"""
import hashlib
from pathlib import Path


def text_digests(path):
    raw=hashlib.sha256(); lf=hashlib.sha256(); crlf=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for line in stream:
            normalized=line.replace(b"\r\n",b"\n")
            raw.update(line); lf.update(normalized); crlf.update(normalized.replace(b"\n",b"\r\n"))
    return {"raw":raw.hexdigest(),"lf":lf.hexdigest(),"crlf":crlf.hexdigest()}


def matches_text_digest(path,expected):
    return expected in text_digests(path).values()
