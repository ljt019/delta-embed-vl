# /// script
# dependencies = [
#   "vllm",
#   "torch-c-dlpack-ext",
# ]
# ///
import os
import subprocess
import sys

GPU_MEMORY_UTILIZATION = "0.95"
DATA_PARALLEL_SIZE = "2"
API_SERVER_COUNT = "5"

COMMAND = [
    "vllm",
    "serve",
    "Qwen/Qwen3-VL-Embedding-8B",
    "--runner",
    "pooling",
    "--convert",
    "embed",
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "8192",
    "--data-parallel-size",
    DATA_PARALLEL_SIZE,
    "--api-server-count",
    API_SERVER_COUNT,
    "--gpu-memory-utilization",
    GPU_MEMORY_UTILIZATION,
]


def main() -> None:
    # Forward any extra args, e.g. --port 8001.
    command = COMMAND + sys.argv[1:]
    env = os.environ | {"VLLM_NO_USAGE_STATS": "1"}
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
