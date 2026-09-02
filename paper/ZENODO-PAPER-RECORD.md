# Zenodo record for the standalone paper upload (paste-ready)

Upload `PAPER.pdf` as its own Zenodo record (upload type **Publication → Preprint**), separate from the
software archive, so the paper gets a citable DOI of its own and the software DOI stays the code's.

**Title**
Reward Hacking in the RL Loop: Oracle-Witnessed Localization and Repair of Reward-Model Exploits During Training

**Authors**
Celekli, Rudy M. — Gradia Research — ORCID 0009-0000-7043-3766

**Description**
Reinforcement learning from human or verifiable feedback has made the reward signal a load-bearing component of
modern post-training, and reward hacking — a policy scoring well on that signal while failing what it was meant to
measure — a central failure mode. This paper carries an oracle-witnessed intervention instrument, introduced for
static benchmark scorers in the Reward-Hacking Wind Tunnel, into the training loop. On a minimal, fully reproducible
task: optimizing against a gameable reward opens a Goodhart gap (proxy 0.98, true 0.00, correlation −0.84) while a
verifiable reward control does not; an exact single-variable fork localizes the exploited feature on the witnessed
sample (lift +1.00, n=64); patching that cue relocates the exploit before a comprehensive patch cures it
(relocation share 0.67); the mechanism spans the implemented policy-gradient and DPO paths; and an online detector
flags hacking before saturation. A frozen, one-seed paired GRPO diagnostic on Qwen2.5-0.5B-Instruct over GSM8K then
shows the exact-match control preserving zero proxy–oracle gap at all 13 evaluations while the gameable arm finishes
at proxy 58/64 versus oracle 1/64 (gap 0.890625); both final adapters reproduce their complete 64-row evaluation
digests under fresh model-backed replay. The result is amplification of an existing seam — not emergence from zero,
a population estimate, or evidence of capability improvement. Code, evidence chains and adapters:
https://github.com/rudycelekli/gradia-reward-loop (Apache-2.0), archived at doi:10.5281/zenodo.22259605.
Pillar 4 of the Gradia program on verifiable evaluation (gradiahq.com).

**Version**  1.0.3          **Publication date**  2026-09-02          **Language**  English

**License**  Creative Commons Attribution 4.0 International (paper text and figures)

**Keywords**
reward hacking; reinforcement learning; RLHF; RLVR; GRPO; DPO; reward models; Goodhart's law;
causal evaluation; reproducibility; AI safety

**Related identifiers**
- `10.5281/zenodo.22259605` — *is supplemented by* (the software and evidence archive)
- `https://github.com/rudycelekli/gradia-reward-loop` — *is supplemented by*
- `10.5281/zenodo.22233638` — *continues* (the Reward-Hacking Wind Tunnel, Pillar 3)
- `10.5281/zenodo.22087020` — *references* (Interruptible Universes, Pillar 1)

**Subjects / communities**  Machine learning; Artificial intelligence safety (join a community only if it fits)

**After publishing**  copy the new paper DOI into `CITATION.cff` (`preferred-citation.doi`) and the README
citation block, then cut v1.0.4 so the software record points at the paper record too.
