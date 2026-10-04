"""Rules and mechanics of tools/make_submission_zip.py (the tool and this test are not part of an anonymized export).

The strings below are the identifying strings of this repository. The tests show that each of them is replaced, that
ordinary words containing them are left alone, that the final search finds what a rule would miss, and that no hash
of an original file survives: a hash of a file that contained the account name would let a reader test guesses.
"""
import gzip
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/make_submission_zip.py"
if not TOOL.is_file():
    pytest.skip("anonymized export: the export tool is not included", allow_module_level=True)
spec = importlib.util.spec_from_file_location("make_submission_zip", TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
RULES = tool.Rules()
HOME = "/home/" + "omote"
KEY = b"k" * 32


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sub(text, name="notes.txt"):
    counts = {}
    return RULES.substitute(name, text, counts), counts


def test_replacements():
    cases = {
        HOME + "/reprise_p4_20260928/code": "/home/anonymous/reprise_p4_20260928/code",
        HOME + "/granood_ke/.venv/bin/python": "/home/anonymous/shared_env/.venv/bin/python",
        "ps -u omote -o args=": "ps -u anonymous -o args=",
        "drwxr-xr-x 2 omote omote 4096": "drwxr-xr-x 2 anonymous anonymous 4096",
        '"stdout": "total 8\\nomote 4242"': '"stdout": "total 8\\nanonymous 4242"',            # after an escaped newline
        "\x1b[34momote\x1b[0m": "\x1b[34manonymous\x1b[0m",                                    # after a colour code
        "https://github.com/omotes24/SalmonLadder": "https://github.com/anonymous/SalmonLadder",
        "-e git+https://github.com/omotes24/MAF07.git@3c96855d#egg=maf07": "-e git+https://github.com/anonymous/unrelated-project",
        "set by: user (Kotaro)": "set by: user ([author])",
        "KotaroOmote <name@keio.jp>": "[author] <name@institution.jp>",
        "OmoteKotaro": "[author]",
        "Koutarou Omote, Kotarou, Kohtaro, Kōtarō, Kôtarô, Kootaroo": "[author], [author], [author], [author], [author], [author]",
        "ssh host.campus.keio.ac.jp": "ssh server.example.org",
        "HEAD was 965e74420e6395cda902486aa62ed8fe04d6bcae": "HEAD was 0000000",
        "commit 965e744": "commit 0000000",
        "Claude-Session: https://claude.ai/code/session_0123abc": "Claude-Session: [link removed]",
        "d-hacks B3, d_hacks, dhacks, d.hacks": "[lab] B3, [lab], [lab], [lab]",
    }
    for before, after in cases.items():
        assert sub(before)[0] == after, before
    for unchanged in ("used to promote peace", "remote sensing", "the shades of grey", "locomote", "and hacks around it", "transfer",
                      "sha256 aa965e74420e6395cda902486aa62ed8fe04d6bcae11"):
        assert sub(unchanged) == (unchanged, {}), unchanged


def test_maintainer_lines_leave_markdown_only():
    text = "keep\nMaintainers: see docs <!-- maintainers -->\nkeep too\n"
    assert sub(text, "README.md")[0] == "keep\nkeep too\n"
    assert sub(text, "notes.txt")[0] == text


def test_final_search_sees_inside_files():
    leak = HOME + "/x"
    assert tool.search("a/b.txt", leak.encode(), RULES)
    assert tool.search("a/" + "omote" + "/b.txt", b"nothing", RULES)                         # in the path
    assert tool.search("a/b.txt.gz", gzip.compress(leak.encode()), RULES)
    assert tool.search("a/b.bin", gzip.compress(leak.encode()), RULES)                       # gzip without the suffix
    buffer = io.BytesIO()
    np.savez_compressed(buffer, paths=np.array([leak, "ok"]), x=np.arange(3))
    assert tool.search("a/b.npz", buffer.getvalue(), RULES)
    buffer = io.BytesIO()
    np.save(buffer, np.array([leak.encode()]))
    assert tool.search("a/b.npy", buffer.getvalue(), RULES)
    buffer = io.BytesIO()
    np.save(buffer, np.array([(1, leak)], dtype=[("n", "i4"), ("path", "U40")]))            # structured array
    assert tool.search("a/s.npy", buffer.getvalue(), RULES)
    buffer = io.BytesIO()
    np.save(buffer, np.array([leak, 1], dtype=object), allow_pickle=True)                   # pickled objects are not opened
    with pytest.raises(tool.Abort):
        tool.search("a/o.npy", buffer.getvalue(), RULES)
    with pytest.raises(tool.Abort):
        tool.search("a/b.bin", b"\x00\x01\x02 unknown binary format", RULES)
    for encoded in (leak.encode("utf-16-le"), b"x" + leak.encode("utf-16-be"), leak.encode("utf-32-le")):
        with pytest.raises(tool.Abort):                                                      # not text: refused, inside gzip as well
            tool.search("a/b.gz", gzip.compress(encoded), RULES)
        buffer = io.BytesIO()
        np.save(buffer, np.frombuffer(encoded + b"\0" * (-len(encoded) % 4), dtype=np.uint8))
        assert tool.search("a/w.npy", buffer.getvalue(), RULES), encoded                     # wide characters inside array data
    for encoded in (leak.encode().hex(), __import__("base64").b64encode(b"ab" + leak.encode()).decode()):
        assert tool.search("a/b.txt", ("value: " + encoded).encode(), RULES), encoded        # hex and base64 inside a text file
    for word in ("komote", "omotek", "Omotethe"):                                            # no letter boundary: no rule fires
        assert tool.search("a/b.txt", word.encode(), RULES) == {"account_name": 1}, word
    assert tool.search("a/b.txt", b"someone@example.org", RULES) and tool.search("a/b.txt", b"/home/alice/data", RULES)
    for home in ("\\/home\\/alice\\/data", "C:\\Users\\alice\\data", "C:\\\\Users\\\\alice"):                  # escaped slashes, Windows
        assert "home directory not in allowed_home_dirs" in tool.search("a/b.txt", home.encode(), RULES), home
    png = b"\x89PNG\r\n\x1a\n" + b"".join(len(c).to_bytes(4, "big") + k + c + b"\0\0\0\0" for k, c in ((b"IHDR", b"\0" * 13), (b"IDAT", b"xy"), (b"IEND", b"")))
    assert tool.search("a/logo.png", png, RULES) == {} and tool.transform("a/logo.png", png, str.upper) is png
    with pytest.raises(tool.Abort):                                                          # an image with a text chunk is refused
        tool.search("a/logo.png", png[:-12] + (9).to_bytes(4, "big") + b"tEXt" + b"Author\0xy" + b"\0\0\0\0" + png[-12:], RULES)
    with pytest.raises(tool.Abort):
        tool.search("a/logo.png", png + b"appended", RULES)
    assert not tool.search("a/b.txt", b"promote remote shades /home/anonymous/x noreply@anthropic.com", RULES)
    name = "omote"                                                                           # written so that no rule sees it
    written = {"\\u escapes": "".join("\\u%04x" % ord(c) for c in name), "escapes in a JSON string": json.dumps(json.dumps("おもて")),
               "fullwidth letters": "".join(chr(ord(c) + 0xfee0) for c in name), "halfwidth kana": "ｵﾓﾃ",
               "character references": "&#111;mote and &#x6f;mote", "percent-encoding": "%2Fhome%2F" + "%6F" + name[1:],
               "combining marks": "Ko\u0304taro\u0304", "octal escape": "\\157" + name[1:], "braced escape": "\\u{6f}" + name[1:],
               "named escape": "\\N{LATIN SMALL LETTER O}" + name[1:], "zero-width space": name[:2] + "\u200b" + name[2:],
               "soft hyphen": name[:3] + "\u00ad" + name[3:], "quoted-printable": "=6F" + name[1:]}
    for form, text in written.items():
        assert name not in text and tool.search("a/b.txt", text.encode(), RULES), form
    assert not tool.search("a/b.txt", "100%\\u00e9t&eacute; 50%2 \\x41 ｐｒｏｍｏｔｅ".encode(), RULES)       # resolved, nothing found
    with pytest.raises(tool.Abort):
        tool.search("a/b.gz", gzip.compress(b"data") + b"trailing bytes", RULES)             # cannot be read completely
    named = io.BytesIO()
    with gzip.GzipFile(filename="original_name.txt", mode="wb", fileobj=named) as handle:
        handle.write(b"data")
    with pytest.raises(tool.Abort):
        tool.search("a/b.gz", named.getvalue(), RULES)                                       # gzip header with a file name


def test_allowed_hit_is_bound_to_the_file_hash():
    vocabulary = ROOT / "third_party/tins/third_party/openai_clip/bpe_simple_vocab_16e6.txt.gz"
    data = vocabulary.read_bytes()
    assert tool.search(str(vocabulary.relative_to(ROOT)), data, RULES) == {}                # the public vocabulary, byte-identical
    changed = tool.regzip(gzip.decompress(data) + b"\nextra\n")
    assert "campus" in tool.search("x/bpe_simple_vocab_16e6.txt.gz", changed, RULES)        # any other file with the token is refused


def test_gzip_and_parquet_are_rewritten():
    import pyarrow as pa
    import pyarrow.parquet as pq
    leak = HOME + "/data/img.jpg"

    def redact(rel, data):
        counts = {}
        return tool.transform(rel, data, lambda s: RULES.substitute(rel, s, counts)), counts
    new, counts = redact("m.csv.gz", gzip.compress(("path\n" + leak + "\n").encode()))
    assert counts == {"server_home": 1} and gzip.decompress(new) == b"path\n/home/anonymous/data/img.jpg\n"
    assert new == redact("m.csv.gz", gzip.compress(("path\n" + leak + "\n").encode()))[0]                 # deterministic
    table = pa.table({"path": [leak, "ok"], "nested": [[leak], []], "n": [1, 2], "pair": [{"a": leak, "b": 1}, {"a": "x", "b": 2}]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    new, counts = redact("t.parquet", buffer.getvalue())
    assert counts == {"server_home": 3}
    back = pq.read_table(io.BytesIO(new))
    assert back.schema.equals(table.schema)
    assert back.column("path").to_pylist() == ["/home/anonymous/data/img.jpg", "ok"]
    assert back.column("nested").to_pylist() == [["/home/anonymous/data/img.jpg"], []]
    assert back.column("pair").to_pylist()[0]["a"] == "/home/anonymous/data/img.jpg" and back.column("n").to_pylist() == [1, 2]
    assert not tool.search("t.parquet", new, RULES) and tool.search("t.parquet", buffer.getvalue(), RULES)
    # the strings of a column are treated in one call; a rule that would match across two of them gets them one by one
    crossing = io.BytesIO()
    pq.write_table(pa.table({"note": ["see https://claude.ai/code/abc", "kept " + leak, "end"]}), crossing)
    with pytest.raises(tool.Crossed):
        tool.transform("c.parquet", crossing.getvalue(), lambda s: RULES.substitute("c.parquet", s, {}), joined=True)
    final, report = tool.anonymize({"c.parquet": crossing.getvalue()}, RULES, KEY)
    assert pq.read_table(io.BytesIO(final["c.parquet"])).column("note").to_pylist() == ["see [link removed]", "kept /home/anonymous/data/img.jpg", "end"]
    assert report["replaced"] == {"c.parquet": {"session_link": 1, "server_home": 1}}
    joined = tool.transform("t.parquet", buffer.getvalue(), lambda s: RULES.substitute("t.parquet", s, {}), joined=True)
    assert pq.read_table(io.BytesIO(joined)).equals(back)
    odd = io.BytesIO()
    pq.write_table(pa.table({"s": ["a"]}).replace_schema_metadata({"blob": b"\xff\xfe\x00binary"}), odd)
    assert redact("o.parquet", odd.getvalue())[0] is odd.getvalue() or redact("o.parquet", odd.getvalue())[0] == odd.getvalue()
    untouched = io.BytesIO()
    pq.write_table(pa.table({"n": [1, 2], "s": ["a", "b"]}), untouched)
    data = untouched.getvalue()
    assert redact("u.parquet", data)[0] is data                                                            # unchanged files keep their bytes
    binary = io.BytesIO()
    np.save(binary, np.frombuffer(leak.encode() + b"\0" * (-len(leak) % 4), dtype=np.uint8))
    binary = binary.getvalue()
    assert redact("b.npy", binary)[0] is binary and tool.search("b.npy", binary, RULES)                    # binary: not rewritten, but found


def _archive():
    """A small archive with the structures of the real one: code with a server path, records that quote hashes of
    files and of each other, a hash of a server table, public hashes, a gzip manifest, and a provenance record."""
    code = ('HOME = "' + HOME + '"\n').encode()
    table_hash, weights_hash, image_hash = sha(b"server table with paths"), sha(b"public weights"), sha(b"image pixels")
    record = json.dumps({"code_sha256": sha(code), "table_sha256": table_hash, "weights": weights_hash, "table_short": table_hash[:10],
                         "code_md5": hashlib.md5(code).hexdigest()[:12], "code_md5_short": hashlib.md5(code).hexdigest()[:11]}, indent=1).encode()
    manifest_text = ('{"path": "' + HOME + '/data/a.jpg", "sha256": "' + image_hash + '"}\n').encode()
    manifest = tool.regzip(manifest_text)
    files = {
        "experiments/x/code.py": code,
        "experiments/x/record.json": record,
        "experiments/x/record.sha256": (sha(record) + "  record.json\n").encode(),
        "experiments/x/later.json": json.dumps({"record": sha(record), "code": "see " + sha(code)[:12], "table": table_hash,
                                                "manifest_text_sha256": sha(manifest_text), "image": image_hash}).encode(),
        "experiments/x/run.sh": ("# table " + table_hash + "\n").encode(),
        "experiments/x/untouched.json": json.dumps({"n": 1, "seed": 20260927}).encode(),
        "third_party/y/clip.py": ('URL = "https://example.org/' + weights_hash + '/model.pt"\n').encode(),
        "manifests/reference_manifest.jsonl.gz": manifest,
    }
    listed = {rel: {"sha256": sha(data), "bytes": len(data)} for rel, data in files.items()}
    tables = {"manifests/reference_manifest.jsonl.gz": {"sha256": sha(manifest), "bytes": len(manifest), "columns": {"path": "str"},
                                                         "server_parquet_sha256": table_hash, "server_parquet_bytes": 4162180}}
    files["provenance/server_snapshot.json"] = json.dumps({"files": listed, "tables": tables}, indent=1).encode()
    inputs = [{"path": HOME + "/x/code.py", "bytes": len(code), "sha256": sha(code)},                    # archived, listed by server path
              {"path": HOME + "/x/table.parquet", "bytes": 4162180, "sha256": table_hash},               # not archived
              {"path": "weights.pt", "bytes": 1234, "sha256": weights_hash}]                             # public
    files["experiments/x/inputs.json"] = json.dumps({"inputs": inputs, "bytes": 77}, indent=2).encode()
    files["experiments/x/inputs.jsonl"] = "".join(json.dumps(entry) + "\n" for entry in inputs).encode() + b"not json\n"
    # a changed file whose hash begins with eight decimal digits: a record that holds the same digits as a number keeps them
    numbered = next(c for c in ((HOME + " %d\n" % n).encode() for n in range(10 ** 6)) if sha(c)[:8].isdigit() and sha(c)[0] != "0")
    files["experiments/x/numbered.txt"] = numbered
    files["experiments/x/counts.json"] = json.dumps({"n": int(sha(numbered)[:8]), "numbered_sha256": sha(numbered)}).encode()
    return files, {"table": table_hash, "weights": weights_hash, "image": image_hash, "manifest_text": manifest_text}


def test_no_hash_of_an_original_survives():
    files, values = _archive()
    final, report = tool.anonymize(files, RULES, KEY)
    code, record, later = final["experiments/x/code.py"], json.loads(final["experiments/x/record.json"]), json.loads(final["experiments/x/later.json"])
    assert code == b'HOME = "/home/anonymous"\n'
    # records follow the exported files along the whole chain: code -> record -> hash file and later record -> provenance
    assert record["code_sha256"] == sha(code)
    assert final["experiments/x/record.sha256"].split()[0].decode() == sha(final["experiments/x/record.json"])
    assert later["record"] == sha(final["experiments/x/record.json"]) and later["code"] == "see " + sha(code)[:12]
    assert later["manifest_text_sha256"] == sha(gzip.decompress(final["manifests/reference_manifest.jsonl.gz"]))
    snapshot = json.loads(final["provenance/server_snapshot.json"])
    for rel, info in snapshot["files"].items():
        assert info == {"sha256": sha(final[rel]), "bytes": len(final[rel])}, rel
    # a hash that cannot be verified inside the archive becomes a pseudonym, the same one everywhere, also in our scripts
    pseudonym = record["table_sha256"]
    assert pseudonym != values["table"] and re.fullmatch("[0-9a-f]{64}", pseudonym) and later["table"] == pseudonym
    assert record["table_short"] == pseudonym[:10]                                           # a shortened quote follows
    # other algorithms follow from 12 digits; shorter quotes are not recognised as hashes (documented limit)
    assert record["code_md5"] == hashlib.md5(code).hexdigest()[:12]
    assert record["code_md5_short"] == hashlib.md5(files["experiments/x/code.py"]).hexdigest()[:11]
    counts = json.loads(final["experiments/x/counts.json"])
    assert counts == {"n": int(sha(files["experiments/x/numbered.txt"])[:8]), "numbered_sha256": sha(final["experiments/x/numbered.txt"])}
    assert final["experiments/x/run.sh"] == ("# table " + pseudonym + "\n").encode()
    # sizes next to hashes: the exported size for a changed file, null for a file outside the archive (its size would tell
    # how long the replaced name is), unchanged otherwise; also in nested objects, under a server path and in JSON lines
    exported_manifest = final["manifests/reference_manifest.jsonl.gz"]
    assert snapshot["tables"]["manifests/reference_manifest.jsonl.gz"] == {
        "sha256": sha(exported_manifest), "bytes": len(exported_manifest), "columns": {"path": "str"},
        "server_parquet_sha256": pseudonym, "server_parquet_bytes": None}
    expected = [{"path": "/home/anonymous/x/code.py", "bytes": len(code), "sha256": sha(code)},
                {"path": "/home/anonymous/x/table.parquet", "bytes": None, "sha256": pseudonym},
                {"path": "weights.pt", "bytes": 1234, "sha256": values["weights"]}]
    assert json.loads(final["experiments/x/inputs.json"]) == {"inputs": expected, "bytes": 77}
    lines = final["experiments/x/inputs.jsonl"].decode().splitlines()
    assert [json.loads(line) for line in lines[:3]] == expected and lines[3] == "not json"
    assert len(code) != len(files["experiments/x/code.py"]) and b"4162180" not in b"".join(final.values())
    # public values stay: they also occur in third-party code or in an image manifest
    assert record["weights"] == values["weights"] and later["image"] == values["image"]
    assert final["third_party/y/clip.py"] is files["third_party/y/clip.py"] and final["experiments/x/untouched.json"] is files["experiments/x/untouched.json"]
    # the attack: no exported file contains a hash, or the first digits of a hash, of an original that was changed
    assert set(report["changed"]) == set(files) - {"third_party/y/clip.py", "experiments/x/untouched.json"}
    assert hashlib.md5(files["experiments/x/code.py"]).hexdigest()[:11].encode() in final["experiments/x/record.json"]
    exported = b"\n".join(final.values()) + gzip.decompress(final["manifests/reference_manifest.jsonl.gz"])
    limits = {hashlib.md5(files["experiments/x/code.py"]).hexdigest(), sha(files["experiments/x/numbered.txt"])}   # the two quotes above
    for rel in report["changed"]:
        for digest in set(tool.digests(files[rel]).values()) - limits:
            assert digest[:8].encode() not in exported, (rel, digest)
    assert values["table"][:8].encode() not in exported
    # the final search accepts the result and refuses an archive in which one original hash was left
    leftovers = tool.traces(report["original_digests"])
    verifiable = {d for data in final.values() for d in tool.digests(data).values()} | report["pseudonyms"] | report["public_hashes"]
    for rel, data in final.items():
        assert tool.search(rel, data, RULES, leftovers, verifiable) == {}, rel
    planted = b'{"old": "' + sha(files["experiments/x/code.py"])[:12].encode() + b'"}'
    assert "hash of the original of a changed file" in tool.search("experiments/x/p.json", planted, RULES, leftovers, verifiable)
    stray = b'{"h": "' + sha(b"anything").encode() + b'"}'
    assert "hash that cannot be verified inside the archive" in tool.search("experiments/x/q.json", stray, RULES, leftovers, verifiable)
    # ... also when the hash is cut, wrapped over lines, written with another algorithm, or in base64
    old = sha(files["experiments/x/code.py"])
    piece = "piece of the hash of the original of a changed file"
    assert piece in tool.search("experiments/x/p.txt", ("id " + old[20:40]).encode(), RULES, leftovers, verifiable)
    assert piece in tool.search("experiments/x/p.txt", ("sha256 = " + old[:30] + "\\\n    " + old[30:]).encode(), RULES, leftovers, verifiable)
    assert piece in tool.search("experiments/x/p.json", json.dumps({"h": old[:9] + "\n" + old[9:18] + " " + old[18:27]}).encode(), RULES, leftovers, verifiable)
    for algorithm in ("md5", "sha1", "sha512", "blake2b"):
        other = hashlib.new(algorithm, files["experiments/x/code.py"]).hexdigest()
        assert tool.search("experiments/x/p.txt", other.encode(), RULES, leftovers, verifiable), algorithm
    based = __import__("base64").b64encode(bytes.fromhex(old)).decode()
    for form in (based, based.rstrip("="), "sha256-" + based, based.replace("+", "-").replace("/", "_")):
        assert tool.search("experiments/x/p.txt", ("integrity: " + form + "\n").encode(), RULES, leftovers, verifiable), form
    assert not tool.search("experiments/x/p.txt", ("id " + sha(code)[20:40] + " " + "0123456789" * 4).encode(), RULES, leftovers, verifiable)
    assert tool.search("experiments/x/cache/" + old + ".txt", b"nothing", RULES, leftovers, verifiable)          # in a file name
    assert piece in tool.search("experiments/x/p.txt", ".".join(old[i:i + 8] for i in range(0, 64, 8)).encode(), RULES, leftovers, verifiable)
    assert piece in tool.search("experiments/x/p.txt", "\n".join("# " + old[i:i + 12] for i in range(0, 60, 12)).encode(), RULES, leftovers, verifiable)
    long = hashlib.sha512(files["experiments/x/code.py"]).hexdigest()
    assert piece in tool.search("experiments/x/p.py", ('H = "' + long[-64:] + '"').encode(), RULES, leftovers, verifiable)    # 64 digits, in a script
    assert piece in tool.search("experiments/x/p.py", ('H = "' + old[32:] + old[:32] + '"').encode(), RULES, leftovers, verifiable)
    for algorithm in ("sha3_256", "blake2s", "sha3_512"):
        other = hashlib.new(algorithm, files["experiments/x/code.py"]).hexdigest()
        assert tool.search("experiments/x/p.py", other.encode(), RULES, leftovers, verifiable), algorithm
    based32 = __import__("base64").b32encode(bytes.fromhex(old)).decode()
    for form in (based32, based32.rstrip("=").lower()):
        assert tool.search("experiments/x/p.txt", ("digest " + form + "\n").encode(), RULES, leftovers, verifiable), form


def test_sizes_are_set_only_beside_rewritten_hashes():
    text = '{"a": {"x_sha256": "AA", "x_bytes": 5, "bytes": 6, "deep": {"sha256": "bb", "bytes" : 7}}, "l": [{"sha256": "cc", "bytes": 8}, 9]}'
    spans = tool.json_objects(text)
    assert [sorted(found) for found in spans] == [["a", "l"], ["bytes", "deep", "x_bytes", "x_sha256"], ["bytes", "sha256"], ["bytes", "sha256"]]
    assert text[slice(*spans[1]["x_bytes"])] == "5" and text[slice(*spans[2]["sha256"])] == '"bb"'
    assert tool.with_sizes(text, {"aa": 50, "bb": None}) == text.replace('"x_bytes": 5', '"x_bytes": 50').replace('"bytes" : 7', '"bytes" : null')
    assert tool.with_sizes(text, {"dd": 1}) == text and tool.with_sizes(text, {}) == text
    for broken in ('{"sha256": "aa", "bytes": 5', '{"sha256": "aa", "bytes": 5} tail', 'sha256 aa "bytes": 5', '{"sha256": "aa", "bytes": "5"}'):
        assert tool.with_sizes(broken, {"aa": 50}) == broken                                 # not JSON, or not a number: left alone


def test_pseudonyms_depend_on_the_key_only():
    files, values = _archive()
    first, second = tool.anonymize(files, RULES, KEY)[0], tool.anonymize(files, RULES, KEY)[0]
    other = tool.anonymize(files, RULES, b"another key" * 3)[0]
    assert first == second
    assert json.loads(first["experiments/x/record.json"])["table_sha256"] != json.loads(other["experiments/x/record.json"])["table_sha256"]
    assert first["experiments/x/code.py"] == other["experiments/x/code.py"]


def test_rules_file_is_consistent():
    raw = json.loads((ROOT / "tools/anonymization_rules.json").read_text(encoding="utf-8"))
    assert {"tools/make_submission_zip.py", "tools/anonymization_rules.json", "CITATION.cff", "docs/SUBMISSION.md", "tests/test_anonymization.py", "paper/**"} <= set(raw["exclude"])
    forbidden = {r["id"] for r in raw["forbidden"]}
    assert {a["forbidden"] for a in raw["allow"]} <= forbidden
    for rule in raw["replace"]:                                # a replacement never produces a forbidden string
        text = rule["replacement"].replace("\\1", "")
        assert not any(m.lower() not in words for _, pattern, words in RULES.forbidden for m in pattern.findall(text)), rule["id"]
