#!/usr/bin/env python3
"""Render paper/PAPER.md to PDF (pandoc -> xelatex). Run: python paper/build_pdf.py

Figures are the matplotlib PNGs in ../figures (recomputable via `make figures`), so no mermaid
step is needed. Kept close to the Wind Tunnel build for a consistent look across the program.
"""
import subprocess
import sys
from shutil import which
from pathlib import Path

PAPER = Path(__file__).resolve().parent
HDR = PAPER / "_header.tex"
HDR.write_text(
    "\\usepackage[export]{adjustbox}\n"
    "\\setkeys{Gin}{max width=\\linewidth,keepaspectratio}\n"
    "\\usepackage{longtable,booktabs,array}\n"
    "\\usepackage{ragged2e}\n\\raggedbottom\n"
    "\\usepackage{newunicodechar,amssymb}\n"
    "\\newunicodechar{→}{\\ensuremath{\\rightarrow}}\n"
    "\\newunicodechar{←}{\\ensuremath{\\leftarrow}}\n"
    "\\newunicodechar{↦}{\\ensuremath{\\mapsto}}\n"
    "\\newunicodechar{γ}{\\ensuremath{\\gamma}}\n"
    "\\newunicodechar{Γ}{\\ensuremath{\\Gamma}}\n"
    "\\newunicodechar{δ}{\\ensuremath{\\delta}}\n"
    "\\newunicodechar{Δ}{\\ensuremath{\\Delta}}\n"
    "\\newunicodechar{θ}{\\ensuremath{\\theta}}\n"
    "\\newunicodechar{β}{\\ensuremath{\\beta}}\n"
    "\\newunicodechar{λ}{\\ensuremath{\\lambda}}\n"
    "\\newunicodechar{φ}{\\ensuremath{\\phi}}\n"
    "\\newunicodechar{κ}{\\ensuremath{\\kappa}}\n"
    "\\newunicodechar{ρ}{\\ensuremath{\\rho}}\n"
    "\\newunicodechar{≈}{\\ensuremath{\\approx}}\n"
    "\\newunicodechar{≤}{\\ensuremath{\\leq}}\n"
    "\\newunicodechar{≥}{\\ensuremath{\\geq}}\n"
    "\\newunicodechar{∈}{\\ensuremath{\\in}}\n"
    "\\newunicodechar{×}{\\ensuremath{\\times}}\n"
    "\\newunicodechar{·}{\\ensuremath{\\cdot}}\n"
    "\\newunicodechar{∧}{\\ensuremath{\\wedge}}\n"
    "\\newunicodechar{∨}{\\ensuremath{\\vee}}\n"
    "\\newunicodechar{¬}{\\ensuremath{\\neg}}\n"
    "\\newunicodechar{√}{\\ensuremath{\\sqrt{}}}\n"
    "\\newunicodechar{✓}{\\ensuremath{\\checkmark}}\n"
    "\\newunicodechar{✗}{\\ensuremath{\\times}}\n"
)


def build():
    out = PAPER / "PAPER.pdf"
    tmp = PAPER / "_PAPER.build.pdf"
    tmp.unlink(missing_ok=True)
    engine = which("xelatex") or which("tectonic")
    if engine is None:
        print("FAIL no xelatex or tectonic renderer found")
        return False
    cmd = ["pandoc", "PAPER.md", "-o", tmp.name,
           f"--pdf-engine={engine}",
           "--from", "markdown+pipe_tables+yaml_metadata_block+tex_math_dollars-raw_html-implicit_figures",
           "--toc", "--toc-depth=2",
           "-V", "geometry:margin=1in", "-V", "fontsize=10pt",
           "-V", "colorlinks=true", "-V", "linkcolor=RoyalBlue", "-V", "urlcolor=RoyalBlue",
           "-H", str(HDR)]
    r = subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True)
    ok = r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 20000
    if ok:
        tmp.replace(out)
    else:
        tmp.unlink(missing_ok=True)
    print(("OK  " if ok else "FAIL"), out.name, (out.stat().st_size if out.exists() else 0), "bytes")
    if not ok:
        print(r.stdout[-1800:]); print("STDERR", r.stderr[-1800:])
    return ok


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
