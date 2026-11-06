import math
from typing import Iterable, Sequence, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F

ENDOCHAT_SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."


def format_endochat_teacher_context(conversations: Sequence[dict]) -> Tuple[str, str]:
    """Build the teacher-forcing text and its answer-free prompt prefix."""
    final_assistant = max(
        (index for (index, item) in enumerate(conversations) if item.get("from") == "gpt"),
        default=-1,
    )
    if final_assistant < 0:
        raise ValueError("conversation has no assistant answer")
    prefix = ENDOCHAT_SYSTEM_PROMPT + "\n\n###"
    full = prefix
    prompt = None
    for (index, item) in enumerate(conversations):
        source = item.get("from")
        role = "Human" if source == "human" else "Assistant"
        value = str(item.get("value", ""))
        if index == final_assistant:
            prompt = full + f" {role}:"
            full = prompt + f" {value}"
        else:
            full += f" {role}: {value}\n###"
    if prompt is None:
        raise ValueError("failed to locate final assistant answer")
    return (full, prompt)


def normalized_progress(step: int, max_steps: int) -> float:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return max(0.0, min(1.0, float(step) / float(max_steps)))


def parse_phase_ratios(raw: str | Iterable[float]) -> Tuple[float, float, float]:
    if isinstance(raw, str):
        values = [float(x.strip()) for x in raw.split(",")]
    else:
        values = [float(x) for x in raw]
    if len(values) != 3 or any((x < 0.0 for x in values)):
        raise ValueError("phase ratios must contain three non-negative values")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("phase ratios must have a positive sum")
    return tuple((x / total for x in values))


def progressive_scales(
    progress: float,
    phase_ratios: str | Iterable[float],
    distill_start: float,
    distill_end: float,
    ce_start: float,
    ce_end: float,
) -> Tuple[float, float, float]:
    """Interpolate the supervised and distillation coefficients with training progress."""
    p = max(0.0, min(1.0, float(progress)))
    (r1, r2, _) = parse_phase_ratios(phase_ratios)
    if p <= r1:
        alpha = 0.0
    elif r2 > 0.0 and p <= r1 + r2:
        alpha = (p - r1) / r2
    else:
        alpha = 1.0
    distill = distill_start + (distill_end - distill_start) * alpha
    ce = ce_start + (ce_end - ce_start) * alpha
    return (float(ce), float(distill), float(alpha))


def pacing_progress(
    progress: float, scheduler: str = "linear", geometric_k: float = 5.0, baby_steps: int = 3
) -> float:
    p = max(0.0, min(1.0, float(progress)))
    name = str(scheduler or "linear").lower()
    if name in {"static", "score_only"}:
        return 0.0
    if name == "linear":
        return p
    if name in {"one_pass", "onepass"}:
        return p
    if name == "root":
        return math.sqrt(p)
    if name in {"geom", "geometric"}:
        k = max(1e-06, float(geometric_k))
        return (math.exp(k * p) - 1.0) / (math.exp(k) - 1.0)
    if name in {"baby", "baby_step"}:
        steps = max(1, int(baby_steps))
        index = min(steps - 1, int(p * steps))
        return float(index + 1) / float(steps)
    raise ValueError(f"unsupported continuous pacing scheduler: {scheduler}")


def curriculum_weights(
    scores: torch.Tensor,
    progress: float,
    min_weight: float,
    max_weight: float,
    end_ratio: float,
    order: str = "asc",
    scheduler: str = "linear",
    geometric_k: float = 5.0,
    baby_steps: int = 3,
    hard_subset: bool = False,
    phase_ratios: str | Iterable[float] = "0.3,0.2,0.5",
) -> torch.Tensor:
    """Map difficulty to bounded sample weights and anneal their bias toward one."""
    if not 0.0 < end_ratio <= 1.0:
        raise ValueError("curriculum end_ratio must be in (0, 1]")
    if min_weight <= 0.0 or max_weight < min_weight:
        raise ValueError("invalid curriculum weight interval")
    score = scores.float().clamp(0.0, 1.0)
    order_name = str(order or "asc").lower()
    if order_name == "asc":
        easiness = 1.0 - score
    elif order_name == "desc":
        easiness = score
    elif order_name == "mixed":
        return torch.ones_like(score)
    else:
        raise ValueError(f"unsupported curriculum order: {order}")
    scheduler_name = str(scheduler or "linear").lower()
    shaped_progress = pacing_progress(
        progress, scheduler=scheduler_name, geometric_k=geometric_k, baby_steps=baby_steps
    )
    if hard_subset or scheduler_name in {"baby", "baby_step", "one_pass", "onepass"}:
        if scheduler_name in {"one_pass", "onepass"}:
            (first, second, _) = parse_phase_ratios(phase_ratios)
            if progress <= first:
                (lower, upper) = (0.0, first)
            elif progress <= first + second:
                (lower, upper) = (first, first + second)
            else:
                (lower, upper) = (first + second, 1.0)
            if order_name == "desc":
                (lower, upper) = (1.0 - upper, 1.0 - lower)
            return ((score >= lower) & (score <= upper)).to(score.dtype)
        if order_name == "desc":
            return (score >= 1.0 - shaped_progress).to(score.dtype)
        return (score <= shaped_progress).to(score.dtype)
    initial = min_weight + (max_weight - min_weight) * easiness
    anneal = max(0.0, 1.0 - shaped_progress / float(end_ratio))
    return 1.0 + anneal * (initial - 1.0)


def global_weighted_mean(
    per_sample: torch.Tensor,
    weights: torch.Tensor | None,
    reduction: str = "ratio",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compensate for DDP gradient averaging using the global reduction denominator."""
    if per_sample.ndim != 1:
        raise ValueError("per_sample must be a one-dimensional tensor")
    if weights is None:
        weights = torch.ones_like(per_sample)
    else:
        weights = weights.to(device=per_sample.device, dtype=per_sample.dtype)
    if weights.shape != per_sample.shape:
        raise ValueError("weights and per_sample must have identical shapes")
    reduction_name = str(reduction or "ratio").lower()
    if reduction_name not in {"ratio", "mean"}:
        raise ValueError("weighted reduction must be 'ratio' or 'mean'")
    local_numerator = torch.sum(per_sample * weights)
    if reduction_name == "mean":
        global_denominator = torch.tensor(
            float(per_sample.numel()), dtype=per_sample.dtype, device=per_sample.device
        )
    else:
        global_denominator = torch.sum(weights).detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    return local_numerator * float(world_size) / global_denominator.clamp_min(eps)


def resample_token_trajectory_by_surface_width(
    values: torch.Tensor, source_widths: Sequence[int], target_widths: Sequence[int]
) -> torch.Tensor:
    """Interpolate at normalized token centers and extend endpoint values outside the range."""
    if values.ndim != 2:
        raise ValueError("values must have shape [tokens, features]")
    if values.size(0) == 0:
        raise ValueError("cannot align an empty teacher trajectory")
    if len(source_widths) != values.size(0):
        raise ValueError("source widths must match the teacher trajectory")
    if not target_widths:
        return values.new_zeros((0, values.size(-1)))
    if any((int(width) <= 0 for width in source_widths)):
        raise ValueError("source surface widths must be positive")
    if any((int(width) <= 0 for width in target_widths)):
        raise ValueError("target surface widths must be positive")
    source_width_tensor = torch.tensor(source_widths, dtype=torch.float32, device=values.device)
    target_width_tensor = torch.tensor(target_widths, dtype=torch.float32, device=values.device)
    source_centers = (
        source_width_tensor.cumsum(0) - 0.5 * source_width_tensor
    ) / source_width_tensor.sum()
    target_centers = (
        target_width_tensor.cumsum(0) - 0.5 * target_width_tensor
    ) / target_width_tensor.sum()
    right = torch.searchsorted(source_centers, target_centers).clamp(max=values.size(0) - 1)
    left = (right - 1).clamp(min=0)
    left_position = source_centers.index_select(0, left)
    right_position = source_centers.index_select(0, right)
    denominator = right_position - left_position
    interpolation = torch.where(
        denominator > 0,
        (target_centers - left_position) / denominator.clamp_min(1e-12),
        torch.zeros_like(target_centers),
    ).clamp(0.0, 1.0)
    left_values = values.index_select(0, left)
    right_values = values.index_select(0, right)
    return torch.lerp(left_values, right_values, interpolation.to(values.dtype)[:, None])


def standardize_logits(logits: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must include token and vocabulary dimensions")
    centered = logits - logits.mean(dim=-1, keepdim=True)
    scale = centered.pow(2).mean(dim=-1, keepdim=True).add(float(eps)).sqrt()
    return centered / scale


def center_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must include token and vocabulary dimensions")
    return logits - logits.mean(dim=-1, keepdim=True)
