#!/bin/sh
# MIRU2025 形式のビルド（pLaTeX + dvipdfmx）．main.tex = preamble_p22.tex + body_p22.tex
set -e
cd "$(dirname "$0")"
cat preamble_p22.tex body_p22.tex > main.tex
if command -v extractbb >/dev/null 2>&1; then for f in figs/*.pdf figs/*.png; do [ -f "$f" ] && extractbb "$f"; done; fi
platex -interaction=nonstopmode main >/dev/null 2>&1 || true
platex -interaction=nonstopmode main >/dev/null 2>&1 || true
platex -interaction=nonstopmode main >/dev/null 2>&1 || true
dvipdfmx -q -f ptex-haranoaji.map main 2>/dev/null
echo "Errors: $(grep -c '^!' main.log)  Overfull: $(grep -c Overfull main.log)  Underfull: $(grep -c Underfull main.log)  LaTeX warnings: $(grep -c 'LaTeX Warning' main.log)  $(pdfinfo main.pdf | grep Pages)"
