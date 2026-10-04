# Preparing the supplementary archive for double-blind review

This file is for the authors. It is not part of the anonymized archive.

## Why a tool is needed

The repository is a faithful archive of the experiment server. It names its origin in many places: the account name in about 150,000 server paths, the citation file, one decision record that names the person who set a criterion, a commit hash of this repository inside the propagation audit, a link to an assistant session inside one patch file, and the git history itself. A zip made by hand or by "Download ZIP" contains all of them.

Replacing these strings is not enough. The archive records the SHA-256 of every file. If a file contained the account name and its original hash stays in the archive, a reader can put a guessed name back into the exported file, hash it, and compare: the name is found in seconds. The same holds for the recorded hashes of server tables that contained paths, because those tables can be rebuilt byte for byte from the manifests. `tools/make_submission_zip.py` therefore rewrites hashes as well, verifies the result, and stops without writing a zip if it finds any trace.

## Building the archive

```bash
git status                      # the working tree must be clean
python tools/make_submission_zip.py --out ../salmon_ladder_supplementary.zip --tests
```

`--tests` also runs all CPU checks inside the export (needs PyTorch). The export takes about four minutes on a laptop-class CPU. The tool prints the hash of the zip and writes `../salmon_ladder_supplementary.zip.receipt.json`. **Upload only the zip. Keep the receipt and never upload it**: it holds the source commit, the key of the pseudonyms, and the original and exported hash of every changed file, which is what links the anonymized archive to this repository after the review.

What the tool does:

1. Takes the files tracked by git and leaves out `CITATION.cff`, `.github/`, this file, the tool, its rules and its test. Symbolic links are refused.
2. Applies the `replace` rules of `tools/anonymization_rules.json` to text files, to text inside gzip files and to strings inside parquet files: server home directory, account names, personal and laboratory names, the name of the shared Python environment, the commit hash of this repository, session links, and the line of the README that points to this file.
3. Rewrites hashes.
   - A hash of an archived file that changed is replaced by the hash of the exported file wherever it stands as a hexadecimal value: complete, or its first digits (from 8 digits for SHA-256, from 12 for other algorithms). This is repeated along the chains of records that quote each other. Inside the export, `tools/verify_snapshot.py` therefore verifies every file against the records, and no record keeps a hash of an original. The registrations and locks that quote such hashes get new hashes themselves.
   - Every other SHA-256 value in a record that cannot be verified inside the archive (server tables, feature files, earlier versions of scripts, unarchived results, public files that are not in the archive) is replaced by a keyed pseudonym. Equal values stay equal, so the documented deviations of the final test still match their registration. Image hashes in the manifests and hashes of public model weights are kept (`keep_hashes` in the rules), and third-party code is not touched.
   - The size that a JSON record gives next to a rewritten hash (`"bytes"` beside `"sha256"`) is set to the size of the exported file, or to `null` for a file that is not part of the archive. The size of a file that contained server paths tells how long the replaced names are.
4. Writes `provenance/anonymized_export.json`, which lists the changed files without any original hash.
5. Searches every exported file and its path for the `forbidden` patterns, for e-mail addresses and home directories that the rules do not list, for hashes that step 3 should have replaced, and for any hash of an original of a changed file.
   - It looks inside gzip, zip, tar, bz2, xz, parquet, npz and npy contents. A PNG image passes only if it has no metadata chunks.
   - It finds the listed names in UTF-16/32, hex and base64 form, and after escapes (`\uXXXX`, octal, `\N{..}`), character references, percent-encoding, quoted-printable, fullwidth and invisible characters are resolved.
   - It finds a hash as its hexadecimal value under the common algorithms, as a prefix, as any 16 digits of it (a value that is cut, grouped or wrapped over lines), and in base64 or base32 form.
   - One hit aborts the export; so does a format the search cannot open. The only allowed hit is a vocabulary token in the public CLIP byte-pair vocabulary, identified by the hash of that file.
6. Runs the verification scripts inside the export, writes the zip with sorted entries and fixed timestamps, reads it back, and compares it with what was searched.

The pseudonyms depend on a random key, so two exports of the same commit differ. `--key-from <receipt>` reuses the key of an earlier export and reproduces its zip on the same machine.

At the commit that introduced the tool the archive holds about 1,960 files in 71 MB; 189 files have replaced strings, 48 further files have updated hashes only, and about 1,250 hashes are pseudonyms.

## Check before uploading

- Unpack the zip in an empty directory and run `python tools/run_cpu_checks.py` there.
- Read `review_before_upload` in the receipt. It lists the files with Japanese text, the e-mail-like strings, the owners of the cited GitHub repositories, and numbers that equal the original size of a changed file and stand near its name (empty in a clean export).
- **Names in kanji** are not in the rules: `names_in_japanese_script` holds kana forms only. Add the kanji forms of the authors, the laboratory and the institution to `forbidden` (and to `replace` if they occur) before the final export.
- The rules leave the following unchanged. Decide for each whether it may stay:
  - **Text in Japanese**: the requests quoted in the Phase 7 registration and its amendments 2 and 3, comments and report strings in scripts of the early stages, and three run logs. An archived file cannot simply be excluded: the verification inside the export would report it as missing. Prefer to keep these files; if one has to go, remove it from the repository and from its provenance record first.
  - **Time zone**: the patch file carries the offset +0900, one amendment says JST, and `docs/PREREGISTRATION.md` explains three typed times as local times (UTC+9).
  - **The server name** `hades` (a file name and several logs). It is a common host name and is kept.
  - **Assistants**: the records show that coding assistants ran the experiments on request (`User (date): ...` entries, a patch authored by an assistant, the word "Codex" in the early registrations). This does not identify the authors; check what the venue asks authors to state about such tools.
  - **U3**: the registration and the aggregated results of the private wildlife set are part of the Phase 4 records and are therefore in the archive, although the paper does not report U3. `docs/RESULTS.md` says so in one sentence. The directory names of that data set (`WILD_DATA2`, `bigcats`) and its species list are kept; add a rule if these names are public elsewhere in connection with the authors.
  - **The working name** REPRISE in the archived files and directory names. It is not the name of the paper, and the first commit of the public repository carries it.
- Do not add the paper sources, figures with names, or the receipt to the zip.

What the tool cannot remove:

- **Lengths.** A replaced name has a different length than the placeholder. Sizes recorded next to hashes in JSON records are updated or removed (step 3). A size in another layout (another member name, a CSV or Markdown table, a log line) is not rewritten; the receipt lists such numbers when they stand near the name of the file. Columns that a log aligned to the width of a path may still tell how many characters a replaced name has. They do not tell which characters.
- **Hashes in other forms.** Not recognised, and therefore neither rewritten nor reported: a quote of fewer than 8 digits of a SHA-256 value or fewer than 12 digits of another algorithm, a quote of 8 to 11 digits that are all decimal, the last digits of a hash when fewer than 16 are given, a hash as raw bytes or as a decimal number, and encodings other than hexadecimal, base64 and base32. Hashes of files outside the archive are replaced only as 64-digit SHA-256 values in records. No record of the archive is known to use one of these forms.
- **Content.** Anyone who has both the zip and this repository can see that they are the same work: file names such as `engine7.py`, the registration texts and the result numbers are identical. The anonymization protects against reading the identity from the zip, not against a search for a public copy.

## The public repository during review

The repository on GitHub is public, carries the account name, and is named after the method, so a search for the name of the method finds it. Do not cite it or link it in the submission. Whether a public repository may stay online during review is decided by the venue. The author guidelines of CVPR 2026 allowed existing public repositories to remain as long as the submission does not point to them; the guidelines of CVPR 2027 were not yet published on 2026-10-04 and have to be checked. Making the repository private until the decision removes the question.

Dates announced for CVPR 2027 on 2026-10-04 (anywhere on Earth): paper registration 10 November 2026, paper submission 16 November 2026, supplementary material 23 November 2026. Verify them at <https://cvpr.thecvf.com/Conferences/2027/Dates> and read the supplementary-material limits (size, format, anonymity of code) in the author guidelines once they are published.

## Changing the rules

`tools/anonymization_rules.json` holds the patterns. A new identifying string needs an entry in `replace` (what to write instead) and in `forbidden` (so that the final search proves that it is gone); a string that should also be found in encoded form goes into `encoded_needles`. An e-mail address or a home directory of a third party that may stay has to be added to `allowed_emails` or `allowed_home_dirs`. Run the tool again; it either produces a zip or names the files that failed the final search. `tests/test_anonymization.py` shows each mechanism on a small synthetic archive, including the attack on the hashes.
