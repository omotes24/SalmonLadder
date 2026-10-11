"""v22: the main reference list (sections/refs22.tex) holds the references cited in the main text (before \\appendix), in first-citation
order; the appendix list (sections/refs_app22.tex, labels A1, A2, ...) holds the references cited only in the appendix, in first-citation
order, taken from sections/refs18.tex or sections/refs_app19_src.tex. Fails on a cited key without a bibitem."""
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
main_part, app_part = src.split("\\appendix", 1)


def cites(text):
    order = []
    for m in re.finditer(r"\\cite\{([^}]+)\}", text):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k not in order:
                order.append(k)
    return order


main_keys = cites(main_part)
app_keys = [k for k in cites(app_part) if k not in main_keys]
refs = (HERE / "sections" / "refs18.tex").read_text(encoding="utf-8")
head = [l for l in refs.splitlines() if not l.startswith("\\bibitem") and "end{thebibliography}" not in l]
items = dict(re.findall(r"^\\bibitem\{([^}]+)\}\s*(.*)$", refs, flags=re.M))
extra = dict(re.findall(r"^\\bibitem\{([^}]+)\}\s*(.*)$", (HERE / "sections" / "refs_app19_src.tex").read_text(encoding="utf-8"), flags=re.M))
allitems = {**extra, **items}
missing = [k for k in main_keys + app_keys if k not in allitems]
if missing:
    sys.exit(f"cited without bibitem: {missing}")
bad = [k for k in main_keys if k not in items]
if bad:
    sys.exit(f"cited in the main text but only in the appendix source: {bad}")
(HERE / "sections" / "refs22.tex").write_text("\n".join(head) + "\n" + "".join(f"\\bibitem{{{k}}} {items[k]}\n" for k in main_keys) + "\\end{thebibliography}\n", encoding="utf-8")
(HERE / "sections" / "refs_app22.tex").write_text("{\\renewcommand{\\refname}{付録の参考文献}\n" + "\n".join(head) + "\n" +
                                                 "".join(f"\\bibitem[A{i + 1}]{{{k}}} {allitems[k]}\n" for i, k in enumerate(app_keys)) +
                                                 "\\end{thebibliography}}\n", encoding="utf-8")
unused = [k for k in allitems if k not in main_keys and k not in app_keys]
print(f"{len(main_keys)} main references, {len(app_keys)} appendix references (A1-A{len(app_keys)}): {app_keys}" + (f"; uncited: {unused}" if unused else ""))
