"""Hash-chained evidence bundles for RL runs -- the Wind Tunnel's reproducibility contract,
reused for the reward loop. Bundled runs write append-only frames linked by a SHA-256 chain and
a self-verifying manifest, so stored summaries can be checked and any frame tamper is detected.
The current committed bundle seals the core demo trajectory; other figures regenerate from pinned
code and seeds. Schema remains compatible with the Wind Tunnel evidence manifest (frames_schema v1).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ZERO = "0" * 64


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_bundle(out_dir, run_id: str, frames: list, summary: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chain = ZERO
    with open(out / "frames.jsonl", "w") as fh:
        for fr in frames:
            fr = dict(fr)
            fr["prev"] = chain
            chain = _sha(chain.encode() + _canon(fr))
            fh.write(json.dumps(fr) + "\n")
    manifest = {
        "schema": "gradia-reward-loop-evidence.v1",
        "frames_schema": "gradia-wind-tunnel-frames.v1",
        "run_id": run_id,
        "n_frames": len(frames),
        "frames_chain_head": chain,
        "summary": summary,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["manifest_sha256"] = _sha(_canon(manifest))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_bundle(bundle_dir) -> dict:
    d = Path(bundle_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    chain = ZERO
    n = 0
    for line in (d / "frames.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        fr = json.loads(line)
        if fr.get("prev") != chain:
            return {"ok": False, "reason": f"chain break at frame {n}", "frames": n}
        chain = _sha(chain.encode() + _canon(fr))
        n += 1
    head_ok = chain == manifest.get("frames_chain_head")
    m2 = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    man_ok = _sha(_canon(m2)) == manifest.get("manifest_sha256")
    return {"ok": bool(head_ok and man_ok and n == manifest.get("n_frames")),
            "frames": n, "frames_chain_head_ok": head_ok, "manifest_self_digest_ok": man_ok}
