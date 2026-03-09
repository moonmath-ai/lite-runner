# Run with: uv run python examples/run_ltx2.py
"""Run config for LTX-2 two-stage text/image-to-video pipeline."""

from functools import lru_cache

# from huggingface_hub import snapshot_download  # noqa: F401
from genai_runner import Metric, Param, Runner


@lru_cache
def _ltx2_dir() -> str:
    # return snapshot_download("Lightricks/LTX-2")
    return "/hf_dir/LTX-2"


@lru_cache
def _gemma_dir() -> str:
    # return snapshot_download("google/gemma-3-12b-it-qat-q4_0-unquantized")
    return "/hf_dir/Gemma"


runner = Runner(
    command="uv run python -m ltx_pipelines.ti2vid_two_stages",
    params=[
        # --- Required paths ---
        Param(
            "checkpoint-path",
            type="path",
            default=lambda: f"{_ltx2_dir()}/ltx-2-19b-dev.safetensors",
            help="LTX-2 model checkpoint",
        ),
        Param(
            "gemma-root",
            type="path",
            default=_gemma_dir,
            help="Gemma text encoder directory",
        ),
        Param(
            "distilled-lora",
            type=["path", "float"],
            labels=["path", "strength"],
            default=lambda: [
                f"{_ltx2_dir()}/ltx-2-19b-distilled-lora-384.safetensors",
                "0.8",
            ],
            help="Distilled LoRA for stage 2",
        ),
        Param(
            "spatial-upsampler-path",
            type="path",
            default=lambda: f"{_ltx2_dir()}/ltx-2-spatial-upscaler-x2-1.0.safetensors",
            help="Spatial upsampler model",
        ),
        # --- Prompt ---
        Param("prompt", help="Text prompt"),
        # Param("negative-prompt", help="Negative prompt"),
        # --- Image conditioning ---
        Param(
            "image",
            type=["path-image", "int", "float"],
            labels=["path", "frame", "strength"],
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
        # --- Guidance ---
        Param(
            "video-cfg-guidance-scale",
            type="float",
            default=3.0,
            help="CFG scale for video prompt adherence",
        ),
        Param(
            "video-stg-guidance-scale",
            type="float",
            default=1.0,
            help="Spatio-Temporal Guidance scale",
        ),
        Param(
            "video-rescale-scale",
            type="float",
            default=0.7,
            help="Rescale to reduce oversaturation",
        ),
        Param("video-skip-step", type="int", default=0, help="Skip step N for video"),
        # --- LiteAttention ---
        Param(
            "attention-type",
            default="lite_attention",
            help="Attention backend override",
        ),
        Param(
            "lite-attention-mode",
            choices=["const", "calib", "load", "disable"],
            default="const",
            help="LiteAttention mode",
        ),
        Param(
            "lite-attention-threshold",
            type="float",
            default=-3.0,
            help="LA skip threshold (log2 scale)",
        ),
        Param(
            "lite-attention-target-error",
            type="float",
            default=0.01,
            help="Target error for LA calibration",
        ),
        Param(
            "lite-attention-metric",
            choices=["L1", "Cossim", "RMSE"],
            default="L1",
            help="Error metric for LA calibration",
        ),
        Param("lite-attention-filename", help="LA calibration save/load filename"),
        Param(
            "lite-attention-disabled-steps",
            type="int",
            default=0,
            help="Initial steps with LA disabled",
        ),
        # --- LiteAttention capture (debug) ---
        Param(
            "lite-attention-capture-path",
            value="$output/la_capture.pt",
            type="path-artifact",
        ),
        Param("lite-attention-capture-blocks", help="Block indices (e.g. '0,10,20')"),
        Param(
            "lite-attention-capture-timesteps",
            help="Timestep indices (e.g. '0,40,80')",
        ),
        Param("lite-attention-capture-heads", help="Head indices (e.g. '0,1,2')"),
        Param("lite-attention-capture-batch", help="Batch indices (e.g. '0')"),
        Param("lite-attention-capture-maps", type="bool"),
        Param(
            "lite-attention-capture-res",
            type="int",
            default=256,
            help="Attention map resolution",
        ),
        Param("lite-attention-capture-stats", type="bool"),
        Param(
            "lite-attention-capture-stages",
            default="1,2",
            help="Stages to capture (e.g. '1,2')",
        ),
        # --- Enhance ---
        Param("enhance-prompt", type="bool", prompt=False, default=False),
    ],
    metrics=[
        Metric("skipped_pct", pattern=r"skipped=([\d.]+)%"),
    ],
    tags=["ltx2"],
)

if __name__ == "__main__":
    runner.run()
