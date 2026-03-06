# /// script
# dependencies = [
#   "vllm",
# ]
# ///
import subprocess
import sys

GPU_MEMORY_UTILIZATION = "0.95"
DATA_PARALLEL_SIZE = "2"

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
    "--data-parallel-size",
    DATA_PARALLEL_SIZE,
    "--gpu-memory-utilization",
    GPU_MEMORY_UTILIZATION,
]


def main() -> None:
    # Forward any extra args, e.g. --port 8001.
    command = COMMAND + sys.argv[1:]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
