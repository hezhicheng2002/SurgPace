import logging
from pathlib import Path
import sys
import torch.distributed as dist

root_logger = None


def print_rank0(*args):
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    if local_rank == 0:
        print(*args)


def logger_setting(save_dir=None):
    global root_logger
    if root_logger is not None:
        return root_logger
    else:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)
        if save_dir:
            directory = Path(save_dir)
            directory.mkdir(parents=True, exist_ok=True)
            save_file = directory / "log.txt"
            fh = logging.FileHandler(save_file, mode="a")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
            return root_logger
        return root_logger


def log(*args):
    global root_logger
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    if local_rank == 0:
        logger_setting().info(*args)


def log_trainable_params(model):
    total_params = sum((p.numel() for p in model.parameters()))
    total_trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    log(f"Total Parameters: {total_params}, Total Trainable Parameters: {total_trainable_params}")
    log(f"Trainable Parameters:")
    for (name, param) in model.named_parameters():
        if param.requires_grad:
            print_rank0(f"{name}: {param.numel()} parameters")
