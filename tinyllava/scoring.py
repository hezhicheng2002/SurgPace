from collections import Counter, defaultdict
import torch


def raw_scores(logits):
    """Measure confidence difficulty and dispersion around a re-lifted Lorentz center."""
    probabilities = torch.softmax(logits, dim=-1)
    confidence = 1.0 - probabilities.max(dim=-1).values.mean().item()
    spatial_norm = torch.linalg.vector_norm(logits, dim=-1, keepdim=True)
    u = logits / torch.sqrt(1.0 + spatial_norm.square() + 1e-06)
    x = torch.cat((torch.sqrt(1.0 + u.square().sum(dim=-1, keepdim=True)), u), dim=-1)
    u_mean = u.mean(dim=0, keepdim=True)
    center = torch.cat(
        (torch.sqrt(1.0 + u_mean.square().sum(dim=-1, keepdim=True)), u_mean), dim=-1
    )
    inner = -(x[..., 0] * center[..., 0]) + (x[..., 1:] * center[..., 1:]).sum(dim=-1)
    lorentz = torch.acosh(torch.clamp(-inner, min=1.0 + 1e-06)).mean().item()
    return (float(confidence), float(lorentz))


def assign_task_midrank(records):
    """Assign equal fused scores their average empirical rank within each task."""
    grouped = defaultdict(list)
    for (index, record) in enumerate(records):
        grouped[str(record.get("task_type", "unknown"))].append(index)
    rank_stats = {}
    for (task, indices) in sorted(grouped.items()):
        ordered = sorted(indices, key=lambda index: records[index]["score_min_conf_lorentz_norm"])
        denominator = max(1, len(ordered) - 1)
        start = 0
        counts = Counter((records[index]["score_min_conf_lorentz_norm"] for index in ordered))
        while start < len(ordered):
            value = records[ordered[start]]["score_min_conf_lorentz_norm"]
            end = start + 1
            while (
                end < len(ordered) and records[ordered[end]]["score_min_conf_lorentz_norm"] == value
            ):
                end += 1
            rank = (start + end - 1) / 2.0 / denominator
            for position in range(start, end):
                records[ordered[position]]["score_min_conf_lorentz_midrank_task"] = rank
            start = end
        rank_stats[task] = {
            "count": len(indices),
            "unique_scores": len(counts),
            "maximum_tie_count": max(counts.values()),
        }
    return rank_stats
