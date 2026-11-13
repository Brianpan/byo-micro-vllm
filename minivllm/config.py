"""Engine-wide configuration (mirrors micro_vllm/config.py, flattened)."""

from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Every knob the engine, scheduler, cache and runner share.

    model:   HF model id ("gpt2") or a local directory saved with save_pretrained.
    num_blocks: number of physical KV blocks. ``None`` lets the ModelRunner
             size the cache from free GPU memory (Step 12).
    """

    model: str = "gpt2"
    device: str = "cuda"
    dtype: str = "float32"  # "float32" | "float16" | "bfloat16"

    # KV cache
    block_size: int = 16
    num_blocks: int | None = None
    gpu_memory_utilization: float = 0.9
    enable_prefix_cache: bool = False

    # Scheduler
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 512
    max_model_len: int = 1024

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_num_seqs <= 0 or self.max_num_batched_tokens <= 0:
            raise ValueError("scheduler limits must be positive")
        if self.num_blocks is not None and self.num_blocks < 2:
            raise ValueError("num_blocks must be >= 2 (one block is the null block)")
