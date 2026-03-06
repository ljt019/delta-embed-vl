# /// script
# dependencies = [
#   "vllm",
# ]
# ///
import subprocess
import sys

DEFAULT_GPU_MEMORY_UTILIZATION = "0.4"

COMMAND = [
    "vllm",
    "serve",
    "Qwen/Qwen3-VL-Embedding-8B",
    "--runner",
    "pooling",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--max-model-len",
    "8192",
    "--gpu-memory-utilization",
    DEFAULT_GPU_MEMORY_UTILIZATION,
]


def main() -> None:
    # Forward any extra args, e.g. --port 8001.
    command = COMMAND + sys.argv[1:]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
