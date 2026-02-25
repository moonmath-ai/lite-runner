# /// script
# requires-python = ">=3.11"
# ///
#
# In a real project, your run_config.py would have:
#   dependencies = ["genai-runner @ git+https://github.com/YOU/genai-runner"]
# For these examples, run with: uv run --project .. examples/run_example.py
"""Example run config for the fake model."""

from genai_runner import Metric, Output, Param, Runner

runner = Runner(
    command="python examples/fake_model.py",
    params=[
        Param("prompt", help="Text prompt for generation"),
        Param("image", type="path", help="Input image",
              value=["examples/fake_input.jpg", "0", "0.8"],
              log_as="image"),
        Param("threshold", type="float", default=-3.2, help="Attention threshold"),
        Param("mode", choices=["calib", "fast", "quality"], default="calib",
              help="Generation mode"),
        Param("seed", type="int", default=42, help="Random seed"),
        Param("output-path", value="$output/video.mp4", log_as="video"),
        Param("debug-output", value="$output/debug.pt", log_as="artifact"),
    ],
    outputs=[
        # Uncontrolled output: fake_model writes model_metadata.json to cwd
        Output("model_metadata.json", log_as="artifact", copy_to="$output/model_metadata.json"),
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
