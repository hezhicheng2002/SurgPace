import math
from typing import Sequence, Union
import torch
import torch.nn.functional as F


def _normalized_context_box(
    box: Union[torch.Tensor, Sequence[float]],
    context_scale: float,
    minimum_width: float,
    minimum_height: float,
    device: torch.device,
):
    if not math.isfinite(float(context_scale)) or context_scale < 1.0:
        raise ValueError("Region zoom context_scale must be finite and at least 1.0")
    coords = torch.as_tensor(box, dtype=torch.float32, device=device).flatten()
    if coords.numel() != 4:
        raise ValueError(f"Region zoom box must contain four values, got {coords.numel()}")
    if not bool(torch.isfinite(coords).all()):
        raise ValueError("Region zoom box contains a non-finite value")
    coords = coords.clamp(0.0, 1.0)
    (x1, y1, x2, y2) = coords.unbind()
    (x_low, x_high) = (torch.minimum(x1, x2), torch.maximum(x1, x2))
    (y_low, y_high) = (torch.minimum(y1, y2), torch.maximum(y1, y2))
    x_center = (x_low + x_high) * 0.5
    y_center = (y_low + y_high) * 0.5
    crop_width = ((x_high - x_low) * context_scale).clamp(min=minimum_width, max=1.0)
    crop_height = ((y_high - y_low) * context_scale).clamp(min=minimum_height, max=1.0)
    x_low = torch.minimum((x_center - crop_width * 0.5).clamp_min(0.0), 1.0 - crop_width)
    y_low = torch.minimum((y_center - crop_height * 0.5).clamp_min(0.0), 1.0 - crop_height)
    return (x_low, y_low, x_low + crop_width, y_low + crop_height)


def append_region_roi_token(
    image_features: torch.Tensor,
    box: Union[torch.Tensor, Sequence[float]],
    roi_grid_features: torch.Tensor = None,
) -> torch.Tensor:
    if image_features.ndim != 2:
        raise ValueError(
            f"image_features must have shape [patches, hidden], got {image_features.shape}"
        )
    grid_features = image_features if roi_grid_features is None else roi_grid_features
    if grid_features.ndim != 2:
        raise ValueError(
            f"roi_grid_features must have shape [patches, hidden], got {grid_features.shape}"
        )
    if grid_features.shape[-1] != image_features.shape[-1]:
        raise ValueError("ROI grid and image sequence hidden sizes must match")
    patch_count = int(grid_features.shape[0])
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Region ROI requires a square patch grid, got {patch_count} patches")
    coords = torch.as_tensor(box, dtype=torch.float32, device=image_features.device).flatten()
    if coords.numel() != 4:
        raise ValueError(f"Region ROI box must contain four values, got {coords.numel()}")
    if not bool(torch.isfinite(coords).all()):
        raise ValueError("Region ROI box contains a non-finite value")
    coords = coords.clamp(0.0, 1.0)
    (x1, y1, x2, y2) = coords.unbind()
    (x_low, x_high) = (torch.minimum(x1, x2), torch.maximum(x1, x2))
    (y_low, y_high) = (torch.minimum(y1, y2), torch.maximum(y1, y2))
    minimum_half_extent = coords.new_tensor(0.5 / grid_size)
    x_center = (x_low + x_high) * 0.5
    y_center = (y_low + y_high) * 0.5
    x_half = torch.maximum((x_high - x_low) * 0.5, minimum_half_extent)
    y_half = torch.maximum((y_high - y_low) * 0.5, minimum_half_extent)
    (x_low, x_high) = ((x_center - x_half).clamp(0.0, 1.0), (x_center + x_half).clamp(0.0, 1.0))
    (y_low, y_high) = ((y_center - y_half).clamp(0.0, 1.0), (y_center + y_half).clamp(0.0, 1.0))
    edges = torch.linspace(
        0.0, 1.0, grid_size + 1, dtype=torch.float32, device=image_features.device
    )
    overlap_x = (torch.minimum(edges[1:], x_high) - torch.maximum(edges[:-1], x_low)).clamp_min(0.0)
    overlap_y = (torch.minimum(edges[1:], y_high) - torch.maximum(edges[:-1], y_low)).clamp_min(0.0)
    weights = overlap_y[:, None] * overlap_x[None, :]
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    grid = grid_features.reshape(grid_size, grid_size, grid_features.shape[-1])
    pooled = (grid * weights.to(image_features.dtype).unsqueeze(-1)).sum(
        dim=(0, 1), dtype=torch.float32
    )
    pooled = (pooled / denominator).to(image_features.dtype)
    return torch.cat((image_features, pooled.unsqueeze(0)), dim=0)


def crop_normalized_region_view(
    image: torch.Tensor, box: Union[torch.Tensor, Sequence[float]], context_scale: float = 1.25
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"image must have shape [channels, height, width], got {image.shape}")
    (_, height, width) = image.shape
    (x_low, y_low, x_high, y_high) = _normalized_context_box(
        box,
        context_scale,
        minimum_width=1.0 / width,
        minimum_height=1.0 / height,
        device=image.device,
    )
    left = max(0, min(width - 1, math.floor(float(x_low) * width)))
    top = max(0, min(height - 1, math.floor(float(y_low) * height)))
    right = max(left + 1, min(width, math.ceil(float(x_high) * width)))
    bottom = max(top + 1, min(height, math.ceil(float(y_high) * height)))
    crop = image[:, top:bottom, left:right].unsqueeze(0)
    return F.interpolate(
        crop, size=(height, width), mode="bicubic", align_corners=False, antialias=True
    ).squeeze(0)


def combine_region_zoom_features(
    global_features: torch.Tensor, zoom_features: torch.Tensor, replace_global: bool = False
) -> torch.Tensor:
    if global_features.ndim != 2 or zoom_features.ndim != 2:
        raise ValueError("Region feature sequences must have shape [patches, hidden]")
    if global_features.shape[-1] != zoom_features.shape[-1]:
        raise ValueError("Global and Region zoom hidden sizes must match")
    if replace_global:
        if global_features.shape[0] != zoom_features.shape[0]:
            raise ValueError("Replacing global features requires equal visual sequence lengths")
        return zoom_features
    return torch.cat((global_features, zoom_features), dim=0)


def fuse_region_zoom_grid(
    global_sequence: torch.Tensor,
    global_grid: torch.Tensor,
    zoom_grid: torch.Tensor,
    box: Union[torch.Tensor, Sequence[float]],
    context_scale: float = 1.25,
) -> torch.Tensor:
    for (name, value) in (
        ("global_sequence", global_sequence),
        ("global_grid", global_grid),
        ("zoom_grid", zoom_grid),
    ):
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [patches, hidden]")
    if global_grid.shape != zoom_grid.shape:
        raise ValueError("Global and Region zoom grids must have identical shapes")
    if global_sequence.shape[-1] != global_grid.shape[-1]:
        raise ValueError("Global sequence and Region grids must share hidden size")
    patch_count = int(global_grid.shape[0])
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError("Region zoom fusion requires square visual grids")
    if global_sequence.shape[0] != patch_count - 1:
        raise ValueError(
            "Region zoom fusion requires the historical grid-with-first-token-dropped global sequence"
        )
    (x_low, y_low, x_high, y_high) = _normalized_context_box(
        box,
        context_scale,
        minimum_width=1.0 / grid_size,
        minimum_height=1.0 / grid_size,
        device=global_grid.device,
    )
    left = max(0, min(grid_size - 1, math.floor(float(x_low) * grid_size)))
    top = max(0, min(grid_size - 1, math.floor(float(y_low) * grid_size)))
    right = max(left + 1, min(grid_size, math.ceil(float(x_high) * grid_size)))
    bottom = max(top + 1, min(grid_size, math.ceil(float(y_high) * grid_size)))
    zoom_map = zoom_grid.transpose(0, 1).reshape(1, zoom_grid.shape[-1], grid_size, grid_size)
    resized_zoom = F.interpolate(
        zoom_map.float(), size=(bottom - top, right - left), mode="bilinear", align_corners=False
    ).to(zoom_grid.dtype)
    resized_zoom = resized_zoom.squeeze(0).permute(1, 2, 0).reshape(-1, zoom_grid.shape[-1])
    grid_indices = (
        torch.arange(patch_count, device=global_grid.device)
        .reshape(grid_size, grid_size)[top:bottom, left:right]
        .reshape(-1)
    )
    historical_mask = grid_indices.gt(0)
    fused = global_sequence.clone()
    fused[grid_indices[historical_mask] - 1] = resized_zoom[historical_mask]
    return fused


def mask_normalized_region_content(
    image: torch.Tensor, box: Union[torch.Tensor, Sequence[float]], fill_value: float = 0.0
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"image must have shape [channels, height, width], got {image.shape}")
    coords = torch.as_tensor(box, dtype=torch.float32, device=image.device).flatten()
    if coords.numel() != 4:
        raise ValueError(f"Region mask box must contain four values, got {coords.numel()}")
    if not bool(torch.isfinite(coords).all()):
        raise ValueError("Region mask box contains a non-finite value")
    (_, height, width) = image.shape
    coords = coords.clamp(0.0, 1.0)
    (x1, y1, x2, y2) = coords.unbind()
    (x_low, x_high) = (torch.minimum(x1, x2), torch.maximum(x1, x2))
    (y_low, y_high) = (torch.minimum(y1, y2), torch.maximum(y1, y2))
    left = max(0, min(width - 1, math.floor(float(x_low) * width)))
    top = max(0, min(height - 1, math.floor(float(y_low) * height)))
    right = max(left + 1, min(width, math.ceil(float(x_high) * width)))
    bottom = max(top + 1, min(height, math.ceil(float(y_high) * height)))
    masked = image.clone()
    masked[:, top:bottom, left:right] = float(fill_value)
    return masked
