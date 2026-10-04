"""Build an anonymized archive of this repository for double-blind review (supplementary material).

  python tools/make_submission_zip.py --out ../salmon_ladder_supplementary.zip --tests

The repository is a faithful archive of the experiment server and names its origin. This tool writes a copy from which
the origin cannot be read, and refuses to finish if it finds a trace of it. Replacing strings is not enough: a hash
of a file that contained the account name lets a reader test guesses of that name. The tool therefore also rewrites
hashes. In order:

  1. takes the files tracked by git (clean working tree; --allow-dirty takes untracked files too); symlinks are refused;
  2. leaves out the files listed under "exclude" in tools/anonymization_rules.json;
  3. applies the "replace" rules to text files, to text inside gzip files and to strings inside parquet files;
  4. rewrites hashes, so that no hash of an original that contained a replaced string is left:
       a. a hash (or a prefix of at least 8 hex digits) of an archived file that changed is replaced by the hash of the
          exported file; this is repeated along the chains of records that quote each other, so the export verifies
          against its own records (tools/verify_snapshot.py) and no record keeps a hash of the original;
       b. every other SHA-256 value in a record that cannot be verified inside the archive (hashes of server tables
          with paths, of feature files, of earlier versions of scripts) is replaced by a keyed pseudonym; equal
          values stay equal, also where a script quotes the same value. Files listed under "keep_hashes" (image
          manifests, public model weights) and third-party code are left alone;
       c. the size that a JSON record gives next to a rewritten hash ("...bytes" beside "...sha256") is set to the
          size of the exported file, or to null for a file that is not part of the archive: the size of a file that
          contained server paths tells how long the replaced names are;
  5. writes provenance/anonymized_export.json, which lists the changed files (no original hash, no replaced string);
  6. searches every exported file and path - inside gzip, zip, tar, bz2, xz, parquet, npz and npy contents; in
     UTF-16/32, hex and base64 form; and after escapes, character references, percent-encoding, compatibility and
     invisible characters are resolved - for the "forbidden" patterns, for e-mail addresses and home directories
     that the rules do not allow, for unverifiable hashes that step 4 should have replaced, and for any hash of an
     original of a changed file: the hexadecimal value under the common algorithms, a prefix of it (from 8 digits
     for SHA-256, from 12 for the others), any 16 digits of it (a value that is cut, grouped or wrapped over lines)
     and its base64 and base32 forms. Any hit aborts the export; so does a file that is neither text nor one of
     these formats (a PNG image passes only without metadata chunks). Not recognised: shorter quotes, a hash as raw
     bytes or as a decimal number, and other encodings;
  7. runs the verification scripts inside the export;
  8. writes the zip (sorted entries, fixed timestamps), reads it back and compares it with what was searched;
  9. writes a receipt next to the zip for the authors: source commit, the key of the pseudonyms, and the original
     and exported hash of every changed file. The receipt links the anonymized archive to the original one after
     the review. It is not part of the zip and must not be uploaded.

Requires numpy and pyarrow (requirements-core.txt).
"""
import argparse
import base64
import bz2
import fnmatch
import gzip
import hashlib
import hmac
import html
import io
import json
import lzma
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "tools/anonymization_rules.json"
HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{8,128}(?![0-9a-fA-F])")
NON_LATIN = re.compile("[぀-ヿ㐀-鿿가-힯ｦ-ﾟ]")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
HOME_DIR = re.compile(r"[/\\](?:home|Users)[/\\]+([A-Za-z0-9._-]+)")     # also with backslashes or escaped slashes
GITHUB = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/")
CODE_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".patch", ".cff", ".gitignore", ".gitattributes")
ESCAPE = re.compile(r"\\(?:(\\)|(/)|u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2})|U([0-9a-fA-F]{8})|u\{([0-9a-fA-F]{1,6})\}|([0-3]?[0-7]{1,2})|N\{([^}]{1,80})\})")
QUOTED_PRINTABLE = re.compile(r"=([0-9A-F]{2})")
INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"))      # soft hyphen, zero-width characters
RUN = re.compile(r"[0-9a-fA-F]{16,}")
BASE64 = re.compile(r"[A-Za-z0-9+/_=-]{22,}")
NUMBER = re.compile(r"(?<![0-9A-Za-z._/-])\d{3,}(?![0-9A-Za-z._/-])")
JOIN = {c: None for c in range(128) if not chr(c).isalnum()}       # what may stand between the parts of a wrapped hash
ALGORITHMS = {32: ("md5",), 40: ("sha1",), 56: ("sha224",), 64: ("sha256",), 96: ("sha384",), 128: ("sha512", "blake2b")}
PIECE = 16


class Abort(Exception):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def matches(rel, patterns):
    return any(fnmatch.fnmatchcase(rel, pattern) for pattern in patterns)


# ------------------------------------------------------------------------------------------------ rules
def encoded_forms(needle):
    forms = set()
    for variant in {needle, needle.lower(), needle.upper(), needle.capitalize()}:
        raw = variant.encode()
        forms.update({raw.hex().encode(), raw.hex().upper().encode(), variant.encode("utf-16-le"), variant.encode("utf-16-be"),
                      variant.encode("utf-32-le"), variant.encode("utf-32-be")})
        for pad in range(3):                                # base64 depends on the position of the string in the stream:
            encoded = base64.b64encode(b"\0" * pad + raw)   # keep the characters that the string alone determines
            stable = encoded[[0, 2, 3][pad]:(8 * (pad + len(raw))) // 6]
            if len(stable) >= 6:
                forms.add(stable)
    return forms


class Rules:
    def __init__(self, path=RULES):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.root = raw["archive_root"]
        self.exclude = raw["exclude"]
        self.replace = [(r["id"], re.compile(r["pattern"]), r["replacement"], r.get("only")) for r in raw["replace"]]
        self.forbidden = [(r["id"], re.compile(r["pattern"]), {w.lower() for w in r.get("allowed_words", [])}) for r in raw["forbidden"]]
        self.allow = {(a["forbidden"], a["sha256"]) for a in raw["allow"]}
        self.encoded = {f"needle {i + 1}": re.compile(b"|".join(re.escape(form) for form in sorted(encoded_forms(needle))))
                        for i, needle in enumerate(raw["encoded_needles"])}
        # plain form for binary data, where the text patterns would match noise; needles under 5 bytes would too
        self.plain = re.compile(b"|".join(re.escape(needle.encode()) for needle in raw["encoded_needles"] if len(needle.encode()) >= 5), re.IGNORECASE)
        self.keep_hashes = raw["keep_hashes"]
        self.third_party = raw["third_party"]
        self.allowed_emails = set(raw["allowed_emails"])
        self.allowed_home_dirs = set(raw["allowed_home_dirs"])

    def excluded(self, rel):
        return matches(rel, self.exclude)

    def protected(self, rel):
        """Third-party code and files whose hashes are data (image hashes, public weights): hashes are not rewritten."""
        return matches(rel, self.third_party) or matches(rel, self.keep_hashes)

    def is_record(self, rel):
        return not self.protected(rel) and not rel.endswith(CODE_SUFFIXES)

    def substitute(self, rel, text, counts):
        for rule_id, pattern, replacement, only in self.replace:
            if only and not matches(rel.rsplit("/", 1)[-1], only):
                continue
            text, n = pattern.subn(replacement, text)
            if n:
                counts[rule_id] = counts.get(rule_id, 0) + n
        return text


# ------------------------------------------------------------------------------------------------ file formats
def gunzip(data):
    """Decompressed bytes of a gzip file, or None if the data are not one complete gzip stream."""
    if data[:2] != b"\x1f\x8b":
        return None
    try:
        return gzip.decompress(data)
    except (OSError, EOFError, ValueError):
        return None


def regzip(raw):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as handle:
        handle.write(raw)
    return buffer.getvalue()


def as_text(data):
    """The text of a plain text file, or None (binary data, or NUL bytes: such a file is never rewritten)."""
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def is_container(data):
    return (data[:4] in (b"PK\x03\x04", b"PK\x05\x06") or data[:6] == b"\x93NUMPY" or data[:3] == b"BZh" or data[:6] == b"\xfd7zXZ\x00"
            or data[257:262] == b"ustar")


def walk(value, fn):
    """Apply fn to every string inside a nested python value (parquet cells may be lists, structs or maps)."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [walk(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(walk(v, fn) for v in value)
    if isinstance(value, dict):
        return {k: walk(v, fn) for k, v in value.items()}
    return value


class Crossed(Exception):
    """A function applied to joined strings changed their number: one match covered parts of two strings."""


def batched(values, fn):
    """walk(values, fn) with one call of fn: the strings are joined by NUL characters, which keep matches apart, and
    separated again afterwards. Raises Crossed if that is not the same as treating every string on its own."""
    strings = []
    walk(values, lambda s: strings.append(s) or s)
    if len(strings) < 2 or any("\0" in s for s in strings):
        return walk(values, fn)
    parts = fn("\0".join(strings)).split("\0")
    if len(parts) != len(strings):
        raise Crossed
    parts = iter(parts)
    return walk(values, lambda s: next(parts))


def collect(value, out):
    """Every string or bytes value inside a nested python value, including dict keys."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, (bytes, bytearray)):
        out.append(bytes(value).decode("utf-8", "replace"))
    elif isinstance(value, (list, tuple)):
        for v in value:
            collect(v, out)
    elif isinstance(value, dict):
        for k, v in value.items():
            collect(k, out)
            collect(v, out)


def plain_type(arrow_type):
    import pyarrow as pa
    t = arrow_type
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t) or pa.types.is_temporal(t) or pa.types.is_null(t) or pa.types.is_decimal(t)


def transform(rel, data, fn, joined=False):
    """Apply fn(str) -> str to the text of a file: a text file, the text inside a gzip file, or the strings of a
    parquet file (joined: one call per column, see batched). Returns the original bytes object when nothing changes;
    other formats are never rewritten."""
    if data[:4] == b"PAR1":
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pq.read_table(io.BytesIO(data))
        arrays, changed = list(table.columns), False
        for index, field in enumerate(table.schema):
            if plain_type(field.type):
                continue
            values = table.column(index).to_pylist()
            new_values = batched(values, fn) if joined else walk(values, fn)
            if new_values != values:
                arrays[index], changed = pa.array(new_values, type=field.type), True
        metadata = {k: (fn(as_text(v)).encode("utf-8") if as_text(v) is not None else v) for k, v in (table.schema.metadata or {}).items()}
        if not changed and metadata == (table.schema.metadata or {}):
            return data
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_arrays(arrays, schema=table.schema.with_metadata(metadata)), buffer)
        return buffer.getvalue()
    raw = gunzip(data)
    if raw is not None:
        text = None if is_container(raw) else as_text(raw)
        if text is None:
            return data
        new = fn(text)
        return data if new == text else regzip(new.encode("utf-8"))
    if is_container(data):
        return data
    text = as_text(data)
    if text is None:
        return data
    new = fn(text)
    return data if new == text else new.encode("utf-8")


def views(rel, data, depth=0):
    """(texts, raw byte strings) that a reader could extract from the file. Raises Abort for a format it cannot open."""
    if depth > 4:
        raise Abort(f"{rel}: containers nested too deeply")
    texts, raws = [], [data]
    if data[:4] == b"PAR1":
        import pyarrow.parquet as pq
        handle = pq.ParquetFile(io.BytesIO(data))
        table = handle.read()
        texts.append(table.schema.to_string(show_field_metadata=True, show_schema_metadata=True))
        collect(dict(handle.metadata.metadata or {}), texts)
        cells = []
        for index, field in enumerate(table.schema):
            if not plain_type(field.type):
                collect(table.column(index).to_pylist(), cells)
        try:                                                # dictionary pages can hold values that no row uses
            names = [f.name for f in table.schema if not plain_type(f.type)]
            for column in pq.read_table(io.BytesIO(data), read_dictionary=names).columns:
                for chunk in column.chunks:
                    if hasattr(chunk, "dictionary"):
                        collect(chunk.dictionary.to_pylist(), cells)
        except Exception:                                   # nested columns cannot be read as dictionaries
            pass
        texts.append("\n".join(cells))                      # one text: a line per value
        return texts, raws                                  # the raw bytes cover pages that are stored uncompressed
    raw = gunzip(data)
    if raw is None and data[:2] == b"\x1f\x8b":
        raise Abort(f"{rel}: gzip data that cannot be decompressed completely")
    if raw is None and data[:3] == b"BZh":
        raw = bz2.decompress(data)
    if raw is None and data[:6] == b"\xfd7zXZ\x00":
        raw = lzma.decompress(data)
    if raw is not None:
        if data[:2] == b"\x1f\x8b" and data[3] & 0x1c:
            raise Abort(f"{rel}: gzip header with a file name, comment or extra field")
        inner = views(rel + "!", raw, depth + 1)
        return texts + inner[0], raws + inner[1]
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            texts.append((z.comment or b"").decode("utf-8", "replace"))
            for info in z.infolist():
                texts += [info.filename, (info.comment or b"").decode("utf-8", "replace")]
                inner = views(f"{rel}!{info.filename}", z.read(info), depth + 1)
                texts, raws = texts + inner[0], raws + inner[1]
        return texts, raws
    if data[257:262] == b"ustar":
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for member in t.getmembers():
                texts += [member.name, member.uname, member.gname, member.linkname]
                if member.isfile():
                    inner = views(f"{rel}!{member.name}", t.extractfile(member).read(), depth + 1)
                    texts, raws = texts + inner[0], raws + inner[1]
        return texts, raws
    if data[:6] == b"\x93NUMPY":
        import numpy as np
        try:
            array = np.load(io.BytesIO(data), allow_pickle=False)
        except ValueError:
            raise Abort(f"{rel}: numpy array of pickled objects, which this tool does not open") from None
        texts.append(str(array.dtype))
        names = array.dtype.names or [None]
        for name in names:
            part = array if name is None else array[name]
            if part.dtype.kind == "U":
                texts.append("\n".join(str(x) for x in part.ravel()))
            elif part.dtype.kind == "S":
                texts.append("\n".join(bytes(x).decode("utf-8", "replace") for x in part.ravel()))
        return texts, raws
    if data[:8] == b"\x89PNG\r\n\x1a\n":                     # an image: only the chunks that can hold text are read
        position = 8
        while position + 12 <= len(data):                   # chunk: length, type, content, checksum
            size, kind = int.from_bytes(data[position:position + 4], "big"), data[position + 4:position + 8]
            content = data[position + 8:position + 8 + size]
            if kind not in (b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"pHYs", b"gAMA", b"sRGB", b"cHRM", b"bKGD", b"sBIT"):
                raise Abort(f"{rel}: image with a metadata chunk ({kind.decode('latin-1')}); save the image without metadata")
            position += 12 + size
            if kind == b"IEND":
                break
        if position != len(data):
            raise Abort(f"{rel}: data after the end of the image")
        return texts, raws
    if data[:4] in (b"\x28\xb5\x2f\xfd", b"7z\xbc\xaf", b"Rar!"):
        raise Abort(f"{rel}: compressed format that this tool cannot open")
    text = as_text(data)
    if text is None:
        raise Abort(f"{rel}: binary data in a format that this tool does not open")
    texts.append(text)
    return texts, raws


# ------------------------------------------------------------------------------------------------ hashes
def digests(data):
    """Every hash of a file that a record might quote: of the bytes and, for gzip, of the decompressed bytes."""
    out = {name: hashlib.new(name, data).hexdigest() for name in ("sha256", "md5", "sha1", "sha224", "sha384", "sha512", "blake2b", "blake2s",
                                                                  "sha3_256", "sha3_512")}
    out["git"] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    raw = gunzip(data)
    if raw is not None:
        out.update({name + "-gunzip": hashlib.new(name, raw).hexdigest() for name in ("sha256", "md5", "sha1", "sha512")})
    return out


def usable(token):
    """Short tokens made of decimal digits only are dates, counts or seeds, not prefixes of hashes."""
    return len(token) >= 12 or not token.isdigit()


def prefix_index(pairs):
    """{hex string: value} for every digest and its prefixes: from 8 digits for SHA-256 values (64 hex digits), which is
    how the records of this archive abbreviate hashes, and from 12 digits for the other algorithms."""
    index = {}
    for digest, value in pairs:
        for n in range(8 if len(digest) == 64 else 12, len(digest) + 1):
            index.setdefault(digest[:n], value)
    return index


def traces(original_digests):
    """What the final search looks for of the hashes of the originals of changed files: the value and its prefixes, any
    piece of 16 digits (a value that is cut, grouped or wrapped over lines), and the base64 and base32 forms of the
    complete value."""
    based = set()
    for digest in original_digests:
        raw = bytes.fromhex(digest)
        b32 = base64.b32encode(raw).decode()
        for form in (base64.b64encode(raw).decode(), base64.urlsafe_b64encode(raw).decode(), b32, b32.lower()):
            for text in (form, form.rstrip("=")):
                based.add(text)
                based.update(f"{name}{mark}{text}" for name in ALGORITHMS.get(len(digest), ()) for mark in "-=")
    return {"prefix": prefix_index((d, True) for d in original_digests),
            "piece": {d[i:i + PIECE] for d in original_digests for i in range(len(d) - PIECE + 1)}, "based": based, "cleared": set()}


def unescape(match):
    backslash, slash, u4, x2, u8, braced, octal, named = match.groups()
    if backslash or slash:
        return "\\" if backslash else "/"
    if named:
        try:
            return unicodedata.lookup(named)
        except KeyError:
            return match.group(0)
    return chr(min(int(octal, 8) if octal else int(u4 or x2 or u8 or braced, 16), 0x10ffff))


def readings(text):
    """The text, and what it becomes when the forms in which a name can be written indirectly are resolved: escapes
    (\\uXXXX, \\xXX, \\u{..}, octal, \\N{..}, a doubled backslash, an escaped slash), character references (&#x41;),
    percent-encoding, quoted-printable (=41), compatibility characters (fullwidth and halfwidth forms) and invisible
    characters (soft hyphen, zero width). Repeated, because such forms nest (a JSON string inside a JSON string)."""
    out = [text]
    for _ in range(3):
        new = out[-1]
        if "\\" in new:
            new = ESCAPE.sub(unescape, new)
        if "&" in new:
            new = html.unescape(new)
        if "%" in new:
            new = urllib.parse.unquote(new, errors="replace")
        if "=" in new:
            new = QUOTED_PRINTABLE.sub(lambda m: chr(int(m.group(1), 16)), new)
        if not new.isascii():
            new = unicodedata.normalize("NFKC", new.translate(INVISIBLE))
        if new == out[-1]:
            break
        out.append(new)
    return out


def chunks(text, size=1 << 23):
    """The text in parts of about 8 MB that end at a line break."""
    start = 0
    while start < len(text):
        end = text.find("\n", start + size)
        end = len(text) if end < 0 else end + 1
        yield text[start:end]
        start = end


BLANK = re.compile(r"[ \t\n\r]*")
SCALAR = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|NaN|-?Infinity")
INTEGER = re.compile(r"\d+")


def json_objects(text):
    """The objects of a JSON text, each as {member name: (start, end) of its value in the text}. Raises ValueError,
    IndexError or RecursionError if the text is not one JSON value."""
    objects, string = [], json.decoder.scanstring

    def skip(i):
        return BLANK.match(text, i).end()

    def value(i):
        c = text[i]
        if c == '"':
            return string(text, i + 1)[1]
        if c == "{":
            found = {}
            objects.append(found)
            i = skip(i + 1)
            if text[i] == "}":
                return i + 1
            while True:
                if text[i] != '"':
                    raise ValueError("member name expected")
                key, i = string(text, i + 1)
                i = skip(i)
                if text[i] != ":":
                    raise ValueError("colon expected")
                start = skip(i + 1)
                end = value(start)
                found[key] = (start, end)
                i = skip(end)
                if text[i] == "}":
                    return i + 1
                if text[i] != ",":
                    raise ValueError("comma expected")
                i = skip(i + 1)
        if c == "[":
            i = skip(i + 1)
            if text[i] == "]":
                return i + 1
            while True:
                i = skip(value(i))
                if text[i] == "]":
                    return i + 1
                if text[i] != ",":
                    raise ValueError("comma expected")
                i = skip(i + 1)
        scalar = SCALAR.match(text, i)
        if not scalar or scalar.end() == i:
            raise ValueError("value expected")
        return scalar.end()

    if skip(value(skip(0))) != len(text):
        raise ValueError("text after the JSON value")
    return objects


def with_sizes(text, recorded):
    """Correct the sizes that records give next to rewritten hashes. recorded: {SHA-256 value written by the export:
    size of the exported file, or None for a file that is not part of the archive}. In the JSON object that holds
    "<name>sha256": <value>, the member "<name>bytes" is set to that size, or to null. The text has to be JSON or JSON
    lines; other text is returned unchanged."""
    if not recorded or 'bytes"' not in text:
        return text
    pieces, offset = [], 0
    try:
        pieces = [(0, json_objects(text))]
    except (ValueError, IndexError, RecursionError):
        for line in text.splitlines(keepends=True):
            if 'bytes"' in line:
                try:
                    pieces.append((offset, json_objects(line.rstrip("\r\n"))))
                except (ValueError, IndexError, RecursionError):
                    pass
            offset += len(line)
    edits = []
    for shift, objects in pieces:
        for found in objects:
            for key, (start, end) in found.items():
                partner = found.get(key[:-6] + "bytes") if key.endswith("sha256") else None
                digest = text[shift + start + 1:shift + end - 1].lower()
                if partner and digest in recorded and INTEGER.fullmatch(text, shift + partner[0], shift + partner[1]):
                    size = recorded[digest]
                    edits.append((shift + partner[0], shift + partner[1], "null" if size is None else str(size)))
    for start, end, new in sorted(set(edits), reverse=True):
        text = text[:start] + new + text[end:]
    return text


def like(token, new):
    return new.upper() if token.isupper() else new


def anonymize(files, rules, key):
    """files: {relative path: bytes}. Returns (exported files, report). Pure function of its arguments."""
    counts, base = {}, {}
    for rel, data in files.items():
        for joined in (True, False):                        # False only if a rule matched across two strings of a table
            file_counts = {}
            try:
                base[rel] = transform(rel, data, lambda s: rules.substitute(rel, s, file_counts), joined)
                break
            except Crossed:
                continue
        if file_counts:
            counts[rel] = file_counts
    original = {rel: digests(data) for rel, data in files.items()}
    known = prefix_index(((d, (rel, algo)) for rel, ds in original.items() for algo, d in ds.items()))

    # SHA-256 values in records that cannot be verified inside the archive; values that also occur in protected files stay
    unknown, public = set(), set()
    for rel, data in base.items():
        seen = public if rules.protected(rel) else (unknown if rules.is_record(rel) else None)

        def note(text, seen=seen):
            for token in HEX.findall(text):
                if len(token) == 64 and token.lower() not in known and seen is not None:
                    seen.add(token.lower())
            return text
        transform(rel, data, note, joined=True)
    unknown -= public
    unknown_prefix = prefix_index(((d, d) for d in unknown))

    def pseudonym(digest):
        return hmac.new(key, digest.encode(), hashlib.sha256).hexdigest()

    final, final_digests, visiting, pseudonyms = {}, {}, set(), set()

    def final_digest(rel, algo):
        if rel not in final_digests:
            data = export(rel)
            final_digests[rel] = original[rel] if data is files[rel] or data == files[rel] else digests(data)
        return final_digests[rel][algo]

    def export(rel):
        if rel in final:
            return final[rel]
        if rel in visiting:
            raise Abort(f"{rel}: records quote each other's hashes in a cycle")
        visiting.add(rel)
        data = base[rel]
        if not rules.protected(rel):
            recorded = {}                                   # rewritten SHA-256 value -> size to record next to it

            def swap(match, rel=rel, recorded=recorded):
                token = match.group(0)
                low = token.lower()
                if not usable(token):
                    return token
                hit = known.get(low)
                if hit and hit[0] != rel:
                    new = final_digest(*hit)
                    if new == original[hit[0]][hit[1]]:
                        return token
                    if hit[1] == "sha256" and len(token) == 64:
                        recorded[new] = len(export(hit[0]))  # the file changed: the size of the exported file
                    return like(token, new[:len(token)])
                if low in unknown_prefix:
                    new = pseudonym(unknown_prefix[low])
                    pseudonyms.add(new)
                    if len(token) == 64:
                        recorded[new] = None                # a file outside the archive: its size is not exported
                    return like(token, new[:len(token)])
                return token
            data = transform(rel, data, lambda s: with_sizes(HEX.sub(swap, s), recorded))
        visiting.discard(rel)
        final[rel] = data
        return data

    for rel in files:
        export(rel)
    changed = sorted(rel for rel in files if final[rel] != files[rel])
    report = {"replaced": counts, "changed": changed, "strings_replaced_in": sorted(counts),
              "hashes_updated_in": sorted(set(changed) - set(counts)), "pseudonyms": pseudonyms,
              "mapping": {rel: {"original_sha256": original[rel]["sha256"], "exported_sha256": sha(final[rel])} for rel in changed},
              "public_hashes": public, "original_digests": {d for rel in changed for d in original[rel].values()}}
    return final, report


# ------------------------------------------------------------------------------------------------ the final search
def search(rel, data, rules, leftovers=None, verifiable=None, with_texts=False):
    """{kind of finding: detail} for one exported file; empty if nothing was found.
    leftovers: traces() of the hashes of the originals of changed files; verifiable: hashes that may remain in records."""
    found = {}

    def count(kind, n=1):
        found[kind] = found.get(kind, 0) + n

    digest = sha(data)
    texts, raws = views(rel, data)
    read = [reading for text in texts for reading in readings(text)]
    for text in [rel] + read:
        for rule_id, pattern, allowed_words in rules.forbidden:
            hits = [m for m in pattern.findall(text) if m.lower() not in allowed_words]
            if hits and (rule_id, digest) not in rules.allow:
                count(rule_id, len(hits))
        if not matches(rel, rules.third_party):
            emails = sorted(set(EMAIL.findall(text)) - rules.allowed_emails)
            if emails:
                found["e-mail address not in allowed_emails"] = emails[:5]
        homes = sorted(set(HOME_DIR.findall(text)) - rules.allowed_home_dirs)
        if homes:
            found["home directory not in allowed_home_dirs"] = homes[:5]
    for raw in raws:
        for label, pattern in rules.encoded.items():
            if pattern.search(raw):
                found[f"encoded form (UTF-16/32, hex or base64) of {label} in encoded_needles"] = True
        if as_text(raw) is None and rules.plain.search(raw):
            found["identifying string in binary data"] = True
    if leftovers is not None:
        for text in [rel] + texts:
            tokens = set(HEX.findall(text))
            for token in tokens:
                low = token.lower()
                if usable(token) and low in leftovers["prefix"]:
                    count("hash of the original of a changed file")
                elif len(token) == 64 and rules.is_record(rel) and low not in verifiable and text is not rel:
                    count("hash that cannot be verified inside the archive")
            # a hash that is cut, grouped or wrapped: any run of hex digits, read without what separates its parts,
            # that holds 16 digits of one. Runs that were found clean are not examined again in other files.
            joined = (text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ") if "\\" in text else text).translate(JOIN)
            runs = (set(RUN.findall(joined)) | {t for t in tokens if len(t) >= PIECE}) - leftovers["cleared"]
            for run in runs:
                low = run.lower()
                if any(low[i:i + PIECE] in leftovers["piece"] for i in range(len(low) - PIECE + 1)):
                    count("piece of the hash of the original of a changed file")
                elif len(run) <= 128:
                    leftovers["cleared"].add(run)
            for part in chunks(text):
                if not leftovers["based"].isdisjoint(BASE64.findall(part)):
                    count("base64 or base32 form of the hash of the original of a changed file")
    return (found, read) if with_texts else found


# ------------------------------------------------------------------------------------------------ export
def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def source_files(allow_dirty):
    status = git("status", "--porcelain")
    if status.strip() and not allow_dirty:
        sys.exit("The working tree has uncommitted changes. Commit them, or pass --allow-dirty to export the working tree as it is.")
    links = [line.split("\t", 1)[1] for line in git("ls-files", "-s").splitlines() if line.startswith("120000")]
    if links:
        sys.exit(f"Tracked symbolic links are not exported: {links[:5]}")
    listed = git("ls-files", "-z", "--cached", *(["--others", "--exclude-standard"] if allow_dirty else [])).split("\0")
    listed = sorted({f for f in listed if f and (ROOT / f).is_file()})
    if any((ROOT / f).is_symlink() for f in listed):
        sys.exit("Symbolic links are not exported.")
    return listed, bool(status.strip())


def write_zip(files, modes, root, out):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in sorted(files):
            info = zipfile.ZipInfo(f"{root}/{rel}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if modes.get(rel, 0) & 0o111 else 0o100644) << 16
            z.writestr(info, files[rel], compresslevel=9)


def run_checks(export, with_tests):
    commands = [("verify_snapshot", [sys.executable, "tools/verify_snapshot.py"]),
                ("results_tables", [sys.executable, "tools/make_results_tables.py", "--check"]),
                ("registration_timeline", [sys.executable, "tools/make_registration_timeline.py", "--check"])]
    if with_tests:
        commands.append(("cpu_checks", [sys.executable, "tools/run_cpu_checks.py"]))
    results = {}
    for name, command in commands:
        done = subprocess.run(command, cwd=export, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        results[name] = {"returncode": done.returncode, "tail": (done.stdout + done.stderr).strip().splitlines()[-3:]}
        if done.returncode:
            print(done.stdout + done.stderr)
            sys.exit(f"check '{name}' failed inside the export (tree kept at {export})")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="zip file to write (outside the repository)")
    parser.add_argument("--allow-dirty", action="store_true", help="export the working tree including untracked files")
    parser.add_argument("--tests", action="store_true", help="also run all CPU checks inside the export (needs torch)")
    parser.add_argument("--keep-dir", type=Path, help="keep the exported tree in this (new) directory")
    parser.add_argument("--key-from", type=Path, help="receipt of an earlier export: reuse its key, so that the same commit gives the same zip")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if ROOT == out.parent or ROOT in out.parents:
        parser.error("Write the zip outside the repository.")
    rules = Rules()
    listed, dirty = source_files(args.allow_dirty)
    key = bytes.fromhex(json.loads(args.key_from.read_text())["pseudonym_key"]) if args.key_from else secrets.token_bytes(32)
    files = {rel: (ROOT / rel).read_bytes() for rel in listed if not rules.excluded(rel)}
    modes = {rel: (ROOT / rel).stat().st_mode for rel in files}
    excluded = [rel for rel in listed if rules.excluded(rel)]
    try:
        final, report = anonymize(files, rules, key)
    except Abort as problem:
        sys.exit(f"Nothing was written: {problem}")
    totals = {}
    for file_counts in report["replaced"].values():
        for rule_id, n in file_counts.items():
            totals[rule_id] = totals.get(rule_id, 0) + n
    record = {"description": "This archive is an anonymized export. Account, host and personal names and links were replaced in the files "
                             "listed under strings_replaced_in. Where a file changed, the records that quote its hash were updated to the "
                             "hash of the exported file (hashes_updated_in), along the whole chain of records, and the sizes recorded next "
                             "to these hashes were updated; the hashes of the archived files in the records under provenance/ therefore "
                             "describe the files of this archive, and tools/verify_snapshot.py checks them. SHA-256 values of files that "
                             "are not part of the archive (server tables with paths, feature files, earlier versions of scripts) were "
                             "replaced by pseudonyms, and the sizes recorded next to them by null; equal values remain equal. Image "
                             "hashes in the manifests, hashes of public model weights and third-party files are unchanged. The records "
                             "of checks under provenance/ were written on the complete repository, which has more tests than this "
                             "archive (those of the export itself). The authors keep the correspondence between the original hashes "
                             "and the hashes of this archive.",
              "strings_replaced_in": report["strings_replaced_in"], "hashes_updated_in": report["hashes_updated_in"],
              "pseudonymized_hashes": len(report["pseudonyms"]), "rules_applied": sorted(totals), "excluded_files": len(excluded)}
    final["provenance/anonymized_export.json"] = (json.dumps(record, indent=1, sort_keys=True) + "\n").encode("utf-8")

    leftovers = traces(report["original_digests"])
    verifiable = {d for data in final.values() for d in digests(data).values()} | report["pseudonyms"] | report["public_hashes"]
    old_sizes = {}                                           # original size (as text) -> names of the changed files of that size
    for rel in report["changed"]:
        if len(files[rel]) != len(final[rel]):
            old_sizes.setdefault(str(len(files[rel])), set()).add(rel.rsplit("/", 1)[-1])
    problems, non_latin, emails, owners, sizes_left = {}, [], set(), set(), []
    for rel, data in final.items():
        try:
            found, texts = search(rel, data, rules, leftovers, verifiable, with_texts=True)
        except Abort as problem:
            found, texts = {"unreadable": str(problem)}, []
        if found:
            problems[rel] = found
        if any(NON_LATIN.search(text) for text in texts):
            non_latin.append(rel)
        for text in texts:
            emails.update(EMAIL.findall(text))
            owners.update(GITHUB.findall(text))
            if rules.is_record(rel):                        # a size in a layout that step 4c does not rewrite
                for number in NUMBER.finditer(text):
                    names = old_sizes.get(number.group(0), ())
                    if any(name in text[max(0, number.start() - 300):number.end() + 300] for name in names):
                        sizes_left.append(f"{rel}: {number.group(0)}")
    if problems:
        for rel, found in sorted(problems.items())[:60]:
            print(f"  {rel}: {found}")
        sys.exit(f"{len(problems)} exported file(s) failed the final search. Extend tools/anonymization_rules.json; nothing was written.")

    export = (args.keep_dir.expanduser().resolve() if args.keep_dir else Path(tempfile.mkdtemp(prefix="salmon_ladder_export_")))
    if args.keep_dir and export.exists() and any(export.iterdir()):
        parser.error("--keep-dir must be absent or empty.")
    for rel, data in final.items():
        target = export / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755 if modes.get(rel, 0) & 0o111 else 0o644)
    checks = run_checks(export, args.tests)
    for junk in list(export.rglob("__pycache__")) + list(export.rglob(".pytest_cache")):
        shutil.rmtree(junk, ignore_errors=True)
    if {p.relative_to(export).as_posix() for p in export.rglob("*") if p.is_file()} != set(final):
        sys.exit(f"The checks changed the set of files in the export (tree kept at {export}).")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_zip(final, modes, rules.root, out)
    with zipfile.ZipFile(out) as z:                         # what is in the zip is what was searched
        names = z.namelist()
        if sorted(names) != sorted(f"{rules.root}/{rel}" for rel in final) or any(z.read(f"{rules.root}/{rel}") != data for rel, data in final.items()):
            out.unlink()
            sys.exit("The zip does not match the searched files; it was deleted.")
    receipt = {"zip": out.name, "sha256": sha(out.read_bytes()), "bytes": out.stat().st_size, "files": len(final),
               "source_commit": git("rev-parse", "HEAD").strip(), "working_tree_dirty": dirty, "pseudonym_key": key.hex(),
               "excluded": excluded, "replacements": totals, "files_with_replaced_strings": len(report["strings_replaced_in"]),
               "files_with_updated_hashes_only": len(report["hashes_updated_in"]), "pseudonymized_hashes": len(report["pseudonyms"]),
               "checks": checks,
               "review_before_upload": {
                   "files_with_japanese_text": sorted(non_latin),
                   "email_like_strings": sorted(emails),
                   "github_owners_mentioned": sorted(owners),
                   "original_sizes_next_to_file_names": sorted(set(sizes_left)),
                   "note": "Not identifying by the rules of this tool, listed for a last look: text in Japanese (quoted requests in the "
                           "Phase 7 records, comments in early scripts), addresses of third parties, the owners of the cited repositories, "
                           "and numbers that equal the original size of a changed file and stand near its name (they would tell how "
                           "long a replaced name is; the list is empty in a clean export)."},
               "changed_files": report["mapping"]}
    receipt_path = out.with_name(out.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.keep_dir:
        shutil.rmtree(export, ignore_errors=True)
    shown = ("zip", "sha256", "bytes", "files", "files_with_replaced_strings", "files_with_updated_hashes_only", "pseudonymized_hashes", "replacements",
             "working_tree_dirty")
    print(json.dumps({k: receipt[k] for k in shown}, indent=1))
    print(f"receipt (keep it, do not upload it): {receipt_path}")


if __name__ == "__main__":
    main()
