# Changelog

All notable changes to this research artifact. Versions are git tags; each tagged release is
archived on Zenodo under the record at https://doi.org/10.5281/zenodo.22259605.

## 1.0.3 — 2026-09-02

Paper and metadata only. No change to code, evidence chains, adapters, figures or results.

- Paper: artifact availability statement (repository, DOI, license, verification commands) under the
  abstract and in the Reproducibility and Availability section; author affiliation, contact and ORCID.
- Paper: Section 4.2 states explicitly that the relocation share gamma_local runs in the opposite
  direction to the Wind Tunnel's patch-generalization gamma and is not numerically comparable to it.
- Paper: Section 5.5 reports the verifiable-control exploit probabilities Figure 5 actually shows
  (0.01 policy-gradient, 0.13 DPO) instead of "near zero".
- Paper: numbered figure and table captions, running head, TeX Gyre Pagella typography, version
  and date on the title page; table of contents removed.
- Version bumped to 1.0.3 in `pyproject.toml`, `CITATION.cff`, `.zenodo.json`, README.

## 1.0.2 — 2026-09-02

- Zenodo archival metadata (`.zenodo.json`) and DOI binding in README and `CITATION.cff`.

## 1.0.1 — 2026-09-02

- CI: type-check the optional real-policy dependency.

## 1.0.0 — 2026-09-01

- Replay-verified paired GRPO result on Qwen2.5-0.5B-Instruct over GSM8K: exact-match control with
  zero proxy-oracle gap at all 13 evaluations; gameable arm 58/64 proxy versus 1/64 oracle
  (gap 0.890625); both final adapters replay their 64-row evaluation digests exactly.
- Offline instrument: witnessed localization, repair loop, DPO breadth, over-optimization frontier,
  learned reward model, online detector; 55-check property/control suite; paper and figures.
