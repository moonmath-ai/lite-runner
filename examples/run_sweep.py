# /// script
# dependencies = ["wandb", "questionary"]
# requires-python = ">=3.11"
# ///
"""Example sweep: same model, varying threshold and mode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genai_runner import Metric, Param, Runner

runner = Runner(
    command="python examples/fake_model.py",
    params=[
        Param("prompt", help="Text prompt for generation"),
        Param("threshold", type="float", default=-3.2, help="Attention threshold"),
        Param("mode", choices=["calib", "fast", "quality"], default="calib",
              help="Generation mode"),
        Param("seed", type="int", default=42, help="Random seed"),
        Param("output-path", value="$output/video.mp4", log_as="video"),
    ],
    metrics=[
        Metric("skipped_pct", pattern=r"skipped=([\d.]+)%"),
    ],
    tags=["sweep", "threshold"],
)

if __name__ == "__main__":
    for thresh in [-10, -3, -1, 0]:
        print(f"\n{'='*60}")
        print(f"SWEEP: threshold={thresh}")
        print(f"{'='*60}\n")
        runner.run(overrides={"threshold": thresh})
