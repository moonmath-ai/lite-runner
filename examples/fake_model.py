"""Fake model script that mimics a video diffusion model.

Accepts typical flags, creates dummy output files, prints metrics to stdout.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    """Run a fake video generation model for testing."""
    parser = argparse.ArgumentParser(description="Fake video generation model")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output-path", required=True, help="Output video path")
    parser.add_argument("--debug-output", default=None, help="Debug artifact path")
    parser.add_argument(
        "--image", nargs="+", default=None, help="Input image path + optional args"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=-3.2)
    parser.add_argument("--mode", choices=["calib", "fast", "quality"], default="calib")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"[fake_model] prompt={args.prompt!r}")
    print(
        f"[fake_model] seed={args.seed}, threshold={args.threshold}, mode={args.mode}"
    )
    if args.image:
        print(f"[fake_model] image={args.image}")

    for i in range(3):
        time.sleep(0.3)
        print(f"[fake_model] finished part {i + 1}, {(i + 1) * 125}ms")
        if i == 1:
            print("[fake_model] skipped=32.8%", flush=True)
        print(f"[fake_model] stderr message {i}", file=sys.stderr, flush=True)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\x00" * 1024)
    print(f"[fake_model] wrote video: {out}")

    if args.debug_output:
        dbg = Path(args.debug_output)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        dbg.write_bytes(b"\x00" * 512)
        print(f"[fake_model] wrote debug: {dbg}")

    meta = {"prompt": args.prompt, "seed": args.seed, "threshold": args.threshold}
    meta_path = Path("model_metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[fake_model] wrote metadata: {meta_path.resolve()}")

    print("[fake_model] done")


if __name__ == "__main__":
    main()
