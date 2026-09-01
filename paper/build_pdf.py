#!/usr/bin/env python3
"""Render paper/PAPER.md to PDF (pandoc -> xelatex). Run: python paper/build_pdf.py

Figures are the matplotlib PNGs in ../figures (recomputable via `make figures`), so no mermaid
step is needed. Kept close to the Wind Tunnel build for a consistent look across the program.
"""
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
HDR = PAPER / "_header.tex"
HDR.write_text(
    "\\usepackage[export]{adjustbox}\n"
    "\\setkeys{Gin}{max width=\\linewidth,keepaspectratio}\n"
    "\\usepackage{longtable,booktabs,array}\n"
    "\\usepackage{ragged2e}\n\\raggedbottom\n"
    "\\usepackage{newunicodechar}\n"
    "\\newfontfamily\\symfont{DejaVu Sans}\n"
    + "".join("\\newunicodechar{%s}{{\\symfont %s}}\n" % (ch, ch)
              for ch in "→←↦γΓδΔθβλφκρ≈≤≥∈×·∧∨¬√✓✗")
)


def build():
    out = PAPER / "PAPER.pdf"
    cmd = ["pandoc", "PAPER.md", "-o", "PAPER.pdf",
           "--pdf-engine=xelatex",
           "--from", "markdown+pipe_tables+yaml_metadata_block+tex_math_dollars-raw_html-implicit_figures",
           "--toc", "--toc-depth=2",
           "-V", "geometry:margin=1in",
           "-V", "mainfont=DejaVu Serif", "-V", "sansfont=DejaVu Sans",
           "-V", "monofont=DejaVu Sans Mono", "-V", "fontsize=10pt",
           "-V", "colorlinks=true", "-V", "linkcolor=RoyalBlue", "-V", "urlcolor=RoyalBlue",
           "-H", str(HDR)]
    r = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
    ok = out.exists() and out.stat().st_size > 20000
    print(("OK  " if ok else "FAIL"), out.name, (out.stat().st_size if out.exists() else 0), "bytes")
    if not ok:
        print(r.stdout[-1800:]); print("STDERR", r.stderr[-1800:])
    return ok


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
