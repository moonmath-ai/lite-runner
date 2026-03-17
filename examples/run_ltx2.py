#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "huggingface_hub",
#     "lite-runner @ git+https://github.com/moonmath-ai/LiteRunner",
# ]
# ///
"""Run config for LTX-2 two-stage text/image-to-video pipeline."""

from huggingface_hub import snapshot_download

from lite_runner import Metric, Param, Runner

LTX2_DIR = snapshot_download("Lightricks/LTX-2", local_files_only=True)
GEMMA_DIR = snapshot_download(
    "google/gemma-3-12b-it-qat-q4_0-unquantized", local_files_only=True
)

runner = Runner(
    command="uv run python -m ltx_pipelines.ti2vid_two_stages",
    params=[
        # --- Prompt ---
        Param(
            "prompt",
            help="Text prompt",
            default="A beautiful sunset over the ocean",
        ),
        # --- Image conditioning ---
        Param(
            "image",
            type=["path-image", "int", "float"],
            labels=["path", "frame", "strength"],
            default=["images/surfing-cat.jpg", 0, 0.8],
            help="Input image conditioning",
        ),
        # --- Output ---
        Param("output-path", value="$output/video.mp4", type="path-video"),
        # --- Generation params ---
        Param("seed", type="int", default=10, help="Random seed"),
        Param("num-inference-steps", type="int", default=40, help="Denoising steps"),
        Param(
            "height", type="int", default=1024, help="Video height (divisible by 64)"
        ),
        Param("width", type="int", default=1536, help="Video width (divisible by 64)"),
        Param("num-frames", type="int", default=121, help="Number of frames (8k+1)"),
        Param("frame-rate", type="float", default=24.0, help="Frame rate (fps)"),
        # --- Required paths ---
        Param(
            "checkpoint-path",
            type="path",
            value=f"{LTX2_DIR}/ltx-2-19b-dev.safetensors",
        ),
        Param(
            "gemma-root",
            type="path",
            value=GEMMA_DIR,
        ),
        Param(
            "distilled-lora",
            type=["path", "float"],
            labels=["path", "strength"],
            value=[
                f"{LTX2_DIR}/ltx-2-19b-distilled-lora-384.safetensors",
                0.8,
            ],
        ),
        Param(
            "spatial-upsampler-path",
            type="path",
            value=f"{LTX2_DIR}/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        ),
    ],
    metrics=[
        Metric("stage1-time", pattern=r"40/40 \[(\d\d:\d\d)<", type="timedelta"),
    ],
    tags=[],
)

if __name__ == "__main__":
    runner.run()
