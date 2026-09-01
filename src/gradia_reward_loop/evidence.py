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
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def directory_digest(path: Path) -> str:
    rows = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append({
            "path": file.relative_to(path).as_posix(),
            "sha256": _sha(file.read_bytes()),
            "bytes": file.stat().st_size,
        })
    return _sha(_canon(rows))


def write_bundle(out_dir, run_id: str, frames: list, summary: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chain = ZERO
    with open(out / "frames.jsonl", "w") as fh:
        for fr in frames:
            fr = dict(fr)
            fr["prev"] = chain
            chain = _sha(chain.encode() + _canon(fr))
            fh.write(json.dumps(fr, allow_nan=False) + "\n")
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
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
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
    summary = manifest.get("summary", {})
    pair_ok = True
    if "pair_contract_sha256" in summary:
        pair_path = d / "pair-contract.json"
        if not pair_path.exists():
            pair_ok = False
        else:
            pair_ok = _sha(_canon(json.loads(pair_path.read_text()))) == summary[
                "pair_contract_sha256"
            ]
    checkpoint_ok = True
    if "checkpoint_tree_sha256" in summary:
        final = d / "checkpoints" / "final"
        checkpoint_ok = final.is_dir() and directory_digest(final) == summary[
            "checkpoint_tree_sha256"
        ]
    ok = bool(
        head_ok and man_ok and n == manifest.get("n_frames") and pair_ok and checkpoint_ok
    )
    return {
        "ok": ok,
        "frames": n,
        "frames_chain_head_ok": head_ok,
        "manifest_self_digest_ok": man_ok,
        "pair_contract_ok": pair_ok,
        "checkpoint_tree_ok": checkpoint_ok,
    }


def verify_pair(pair_dir) -> dict:
    root = Path(pair_dir)
    paths = {channel: root / channel for channel in ("verifiable", "gameable")}
    bundles = {
        channel: (
            verify_bundle(path)
            if (path / "manifest.json").exists()
            else {"ok": False, "reason": "missing manifest"}
        )
        for channel, path in paths.items()
    }
    manifests = {}
    for channel, path in paths.items():
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifests[channel] = json.loads(manifest_path.read_text())
    contracts = {
        channel: manifest.get("summary", {}).get("pair_contract_sha256")
        for channel, manifest in manifests.items()
    }
    channels = {
        channel: manifest.get("summary", {}).get("channel")
        for channel, manifest in manifests.items()
    }
    paired = (
        len(manifests) == 2
        and contracts.get("verifiable")
        and contracts.get("verifiable") == contracts.get("gameable")
        and channels == {"verifiable": "verifiable", "gameable": "gameable"}
    )
    return {
        "ok": bool(paired and all(bundle.get("ok") for bundle in bundles.values())),
        "pair_contract_match": bool(paired),
        "bundles": bundles,
    }
