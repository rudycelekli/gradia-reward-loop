#!/usr/bin/env python3
"""Render paper/PAPER.md to PDF (pandoc -> xelatex). Run: python paper/build_pdf.py

Figures are the matplotlib PNGs in ../figures (recomputable via `make figures`), so no mermaid
step is needed. Typography: TeX Gyre Pagella at 10pt on a 1in page, numbered figure and table
captions, a running head, no table of contents (it is a paper, not a report). Falls back to the
Latin Modern defaults when the Pagella OpenType files are not installed, so the build never fails
on fonts alone.
"""
import subprocess
import sys
from shutil import which
from pathlib import Path

PAPER = Path(__file__).resolve().parent
HDR = PAPER / "_header.tex"

UNICODE = {
    "→": "\\rightarrow", "←": "\\leftarrow", "↦": "\\mapsto", "γ": "\\gamma", "Γ": "\\Gamma",
    "δ": "\\delta", "Δ": "\\Delta", "θ": "\\theta", "β": "\\beta", "λ": "\\lambda", "φ": "\\phi",
    "κ": "\\kappa", "ρ": "\\rho", "≈": "\\approx", "≤": "\\leq", "≥": "\\geq", "∈": "\\in",
    "×": "\\times", "·": "\\cdot", "∧": "\\wedge", "∨": "\\vee", "¬": "\\neg", "√": "\\sqrt{}",
    "✓": "\\checkmark", "✗": "\\times",
}

HEADER = (
    "\\usepackage[export]{adjustbox}\n"
    "\\setkeys{Gin}{max width=\\linewidth,keepaspectratio}\n"
    "\\usepackage{longtable,booktabs,array}\n"
    "\\usepackage{ragged2e}\n\\raggedbottom\n"
    "\\usepackage{float}\n\\floatplacement{figure}{H}\n"
    "\\usepackage[font=small,labelfont=bf,labelsep=period,skip=6pt]{caption}\n"
    "\\usepackage{microtype}\n"
    "\\usepackage{fancyhdr}\n\\pagestyle{fancy}\n\\fancyhf{}\n"
    "\\renewcommand{\\headrulewidth}{0pt}\n"
    "\\fancyhead[L]{\\footnotesize\\itshape Reward Hacking in the RL Loop}\n"
    "\\fancyhead[R]{\\footnotesize\\itshape Celekli, 2026}\n"
    "\\fancyfoot[C]{\\footnotesize\\thepage}\n"
    "\\fancypagestyle{plain}{\\fancyhf{}\\fancyfoot[C]{\\footnotesize\\thepage}}\n"
    "\\usepackage{titlesec}\n"
    "\\titleformat{\\section}{\\large\\bfseries}{\\thesection}{0.6em}{}\n"
    "\\titleformat{\\subsection}{\\normalsize\\bfseries}{\\thesubsection}{0.6em}{}\n"
    "\\titlespacing*{\\section}{0pt}{1.4em}{0.6em}\n"
    "\\titlespacing*{\\subsection}{0pt}{1.0em}{0.4em}\n"
    "\\usepackage{newunicodechar,amssymb}\n"
    + "".join(f"\\newunicodechar{{{ch}}}{{\\ensuremath{{{tex}}}}}\n" for ch, tex in UNICODE.items())
)


def build():
    HDR.write_text(HEADER)
    out = PAPER / "PAPER.pdf"
    tmp = PAPER / "_PAPER.build.pdf"
    tmp.unlink(missing_ok=True)
    engine = which("xelatex") or which("tectonic")
    if engine is None:
        print("FAIL no xelatex or tectonic renderer found")
        return False
    have_pagella = subprocess.run(["kpsewhich", "texgyrepagella-regular.otf"],
                                  capture_output=True, text=True).stdout.strip() != ""
    fonts = ["-V", "mainfont=TeX Gyre Pagella", "-V", "sansfont=TeX Gyre Heros",
             "-V", "monofont=Latin Modern Mono", "-V", "monofontoptions=Scale=0.88"] if have_pagella else []
    cmd = ["pandoc", "PAPER.md", "-o", tmp.name,
           f"--pdf-engine={engine}",
           "--from", "markdown+pipe_tables+yaml_metadata_block+tex_math_dollars+implicit_figures+table_captions-raw_html",
           "-V", "geometry:margin=1in", "-V", "fontsize=10pt", "-V", "linestretch=1.06",
           "-V", "colorlinks=true", "-V", "linkcolor=RoyalBlue", "-V", "urlcolor=RoyalBlue",
           "-V", "citecolor=RoyalBlue",
           "-H", str(HDR)] + fonts
    r = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
    ok = r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 20000
    if ok:
        tmp.replace(out)
    else:
        tmp.unlink(missing_ok=True)
    print(("OK  " if ok else "FAIL"), out.name, (out.stat().st_size if out.exists() else 0), "bytes",
          "(Pagella)" if have_pagella else "(Latin Modern fallback)")
    if not ok:
        print(r.stdout[-1800:]); print("STDERR", r.stderr[-1800:])
    return ok


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
