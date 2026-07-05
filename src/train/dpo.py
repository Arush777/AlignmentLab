#!/usr/bin/env python3
"""Phase-2 DPO entrypoint scaffold with Hub SFT checkpoint resolution."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DPO training entrypoint. Phase-2 training is intentionally not implemented yet; "
            "this scaffold locks the --sft-ckpt local-or-hub interface."
        )
    )
    parser.add_argument("--config", type=Path, help="DPO experiment YAML.")
    parser.add_argument(
        "--sft-ckpt",
        required=True,
        help="Local SFT checkpoint path or hub:<repo_id> to fetch into ${ALAB_NODE_TMP}/sft_init.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve --sft-ckpt and exit.")
    return parser.parse_args()


def resolve_sft_ckpt(sft_ckpt: str) -> Path:
    if not sft_ckpt.startswith("hub:"):
        return Path(sft_ckpt).expanduser().resolve()

    repo_id = sft_ckpt.removeprefix("hub:")
    if not repo_id:
        raise ValueError("--sft-ckpt hub: reference must include a repo id")
    node_tmp = Path(os.environ.get("ALAB_NODE_TMP") or f"/tmp/alab_{os.environ.get('LSB_JOBID', 'manual')}")
    dest = node_tmp / "sft_init"
    fetch_script = repo_root_from_script() / "scripts" / "fetch_hub_ckpt.sh"
    result = subprocess.run(
        [str(fetch_script), repo_id, str(dest)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    resolved = result.stdout.strip().splitlines()[-1]
    return Path(resolved)


def main() -> int:
    args = parse_args()
    resolved = resolve_sft_ckpt(args.sft_ckpt)
    print(f"resolved_sft_ckpt={resolved}", flush=True)
    if args.dry_run:
        return 0
    raise NotImplementedError("Phase-2 DPO training is not implemented in this Phase-1.5 change.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
