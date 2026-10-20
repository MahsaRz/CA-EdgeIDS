import hashlib
import importlib.util
from pathlib import Path


def module():
    spec=importlib.util.spec_from_file_location("text_digest",Path(__file__).parents[1]/"scripts/text_digest.py")
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def test_generated_text_checks_allow_only_line_ending_conversion(tmp_path):
    m=module(); lf=b'{\n  "split_seed": 42\n}\n'; crlf=lf.replace(b"\n",b"\r\n")
    a=tmp_path/"linux.json"; b=tmp_path/"windows.json"
    a.write_bytes(lf); b.write_bytes(crlf)
    for expected in [hashlib.sha256(lf).hexdigest(),hashlib.sha256(crlf).hexdigest()]:
        assert m.matches_text_digest(a,expected)
        assert m.matches_text_digest(b,expected)
    a.write_bytes(lf.replace(b"42",b"43"))
    assert not m.matches_text_digest(a,hashlib.sha256(crlf).hexdigest())


def test_generated_sample_value_changes_are_rejected(tmp_path):
    m=module(); content=b"source_row_id,IN_BYTES\r\n123,1000\r\n"
    p=tmp_path/"sample.csv"; p.write_bytes(content)
    expected=hashlib.sha256(content).hexdigest()
    p.write_bytes(content.replace(b"\r\n",b"\n"))
    assert m.matches_text_digest(p,expected)
    p.write_bytes(p.read_bytes().replace(b"1000",b"1001"))
    assert not m.matches_text_digest(p,expected)
