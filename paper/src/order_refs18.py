"""Reorder sections/refs18.tex by first citation (main.tex with \input expanded); keeps the header lines; fails on a missing bibitem; drops uncited ones."""
import re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
def expand(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
        m = re.search(r"\\input\{([^}]+)\}", code)
        if m and "refs" not in m.group(1):
            p = HERE / m.group(1)
            p = p if p.suffix == ".tex" else p.with_suffix(".tex")
            out += [code[:m.start()], expand(p), code[m.end():]]
        else:
            out.append(code)
    return "\n".join(out)
src = expand(HERE / "main.tex")
order = []
for m in re.finditer(r"\\cite\{([^}]+)\}", src):
    for k in (x.strip() for x in m.group(1).split(",")):
        if k not in order:
            order.append(k)
refs = (HERE / "sections" / "refs18.tex").read_text(encoding="utf-8")
lines = refs.splitlines()
head = [l for l in lines if not l.startswith("\\bibitem") and "end{thebibliography}" not in l]
items = dict(re.findall(r"^\\bibitem\{([^}]+)\}\s*(.*)$", refs, flags=re.M))
missing = [k for k in order if k not in items]
if missing:
    sys.exit(f"cited without bibitem: {missing}")
unused = [k for k in items if k not in order]
body = "".join(f"\\bibitem{{{k}}} {items[k]}\n" for k in order)
(HERE / "sections" / "refs18.tex").write_text("\n".join(head) + "\n" + body + "\\end{thebibliography}\n", encoding="utf-8")
print(f"{len(order)} references in first-citation order" + (f"; dropped uncited: {unused}" if unused else ""))
