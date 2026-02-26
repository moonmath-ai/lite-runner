# Run with: uv run python examples/run_example.py
"""Example run config for the fake model."""

from genai_runner import Metric, Output, Param, Runner

runner = Runner(
    command="python examples/fake_model.py",
    params=[
        Param("prompt", help="Text prompt for generation"),
        Param(
            "image",
            help="Input image",
            type=["path-image", "float", "float"],
            labels=["path", "start_frame", "strength"],
            default=["examples/fake_input.jpg", "0", "0.8"],
        ),
        Param("threshold", type="float", default=-3.2, help="Attention threshold"),
        Param(
            "mode",
            choices=["calib", "fast", "quality"],
            default="calib",
            help="Generation mode",
        ),
        Param("seed", type="int", default=42, help="Random seed"),
        Param("output-path", value="$output/video.mp4", type="path-video"),
        Param("debug-output", value="$output/debug.pt", type="path-artifact"),
    ],
    outputs=[
        # Uncontrolled output: fake_model writes model_metadata.json to cwd
        Output(
            "model_metadata.json",
            log_as="artifact",
            copy_to="$output/model_metadata.json",
        ),
    ],
    metrics=[
        Metric("skipped_pct", pattern=r"skipped=([\d.]+)%"),
        Metric("part3_ms", pattern=r"finished part 3, ([\d.]+)ms"),
    ],
    tags=["example"],
    env={"FAKE_MODEL_DEBUG": "1"},
)

if __name__ == "__main__":
    runner.run()
