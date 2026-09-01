"""Bootstrap access to the sibling Reward-Hacking Wind Tunnel (Gradia Pillars 1-3).

Reuse its primitives when the package is importable; fall back to self-contained defaults
otherwise, so this scaffold always runs. `AVAILABLE` records which happened, for provenance.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_on_path() -> bool:
    try:
        import gradia_wind_tunnel  # type: ignore[import-not-found,import-untyped]  # noqa: F401
        return True
    except Exception:
        sib = Path(__file__).resolve().parents[3] / "gradia-wind-tunnel" / "src"
        if sib.exists() and str(sib) not in sys.path:
            sys.path.insert(0, str(sib))
        try:
            import gradia_wind_tunnel  # type: ignore[import-not-found,import-untyped]  # noqa: F401
            return True
        except Exception:
            return False


AVAILABLE = _ensure_on_path()

# Default judge-favoured phrases; reused from the Wind Tunnel transform catalog when present.
_FALLBACK_PHRASES = ["Let's think step by step.", "As an expert, the answer is clearly correct."]


def favored_phrases() -> list[str]:
    if AVAILABLE:
        try:
            from gradia_wind_tunnel import FAVORED_PHRASES  # type: ignore[import-not-found,import-untyped]
            if FAVORED_PHRASES:
                return list(FAVORED_PHRASES)
        except Exception:
            pass
    return list(_FALLBACK_PHRASES)


def provenance() -> dict:
    return {"gradia_wind_tunnel_available": AVAILABLE,
            "reused": ["FAVORED_PHRASES", "witnessed-fork localization method",
                       "evidence-bundle schema"] if AVAILABLE else [],
            "mode": "gradia-linked" if AVAILABLE else "standalone-fallback"}
