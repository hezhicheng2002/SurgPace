import os
import socket
from functools import wraps
import torch
import torch.distributed as dist


def _maybe_init_dist():
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1")))
    except Exception:
        (world_size, local_rank) = (1, -1)
    if world_size > 1 and (not dist.is_initialized()):
        backend = "nccl"
        dist.init_process_group(backend=backend)
        if local_rank >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)


def _install_cuda_bf16_triu_compat():
    if not torch.cuda.is_available() or getattr(torch.triu, "_surgpace_bf16_compat", False):
        return False
    original_triu = torch.triu
    try:
        original_triu(torch.zeros((2, 2), device="cuda", dtype=torch.bfloat16))
        return False
    except RuntimeError as error:
        if "not implemented for 'BFloat16'" not in str(error):
            raise

    @wraps(original_triu)
    def compatible_triu(input_tensor, diagonal=0, *, out=None):
        if input_tensor.is_cuda and input_tensor.dtype == torch.bfloat16:
            result = original_triu(input_tensor.float(), diagonal=diagonal).to(torch.bfloat16)
            if out is not None:
                out.copy_(result)
                return out
            return result
        if out is None:
            return original_triu(input_tensor, diagonal=diagonal)
        return original_triu(input_tensor, diagonal=diagonal, out=out)

    compatible_triu._surgpace_bf16_compat = True
    torch.triu = compatible_triu
    print("[SurgPaceCompat] CUDA BF16 triu uses an FP32 mask-construction fallback", flush=True)
    return True
