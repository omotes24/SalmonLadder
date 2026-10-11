"""v19: the main reference list (sections/refs19.tex, from refs18.tex) in first-citation order, and the references cited only in the
Phase 16 appendix (sections/refs_app19_src.tex -> sections/refs_app19.tex, labels A1, A2, ... in first-citation order).
Fails on a cited key without a bibitem; drops uncited ones."""
import re
import sys
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
head = [l for l in refs.splitlines() if not l.startswith("\\bibitem") and "end{thebibliography}" not in l]
items = dict(re.findall(r"^\\bibitem\{([^}]+)\}\s*(.*)$", refs, flags=re.M))
extra = dict(re.findall(r"^\\bibitem\{([^}]+)\}\s*(.*)$", (HERE / "sections" / "refs_app19_src.tex").read_text(encoding="utf-8"), flags=re.M))
missing = [k for k in order if k not in items and k not in extra]
if missing:
    sys.exit(f"cited without bibitem: {missing}")
main = [k for k in order if k in items]
app = [k for k in order if k not in items and k in extra]
(HERE / "sections" / "refs19.tex").write_text("\n".join(head) + "\n" + "".join(f"\\bibitem{{{k}}} {items[k]}\n" for k in main) + "\\end{thebibliography}\n", encoding="utf-8")
app_head = head[:]
(HERE / "sections" / "refs_app19.tex").write_text("{\\renewcommand{\\refname}{付録の参考文献}\n" + "\n".join(app_head) + "\n" +
                                                 "".join(f"\\bibitem[A{i + 1}]{{{k}}} {extra[k]}\n" for i, k in enumerate(app)) +
                                                 "\\end{thebibliography}}\n", encoding="utf-8")
print(f"{len(main)} main references, {len(app)} appendix references (A1-A{len(app)})" +
      (f"; dropped uncited: {[k for k in items if k not in order] + [k for k in extra if k not in order]}" if any(k not in order for k in list(items) + list(extra)) else ""))
