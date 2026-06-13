"""
Improved loss helpers for SMPL-X / avatar fitting.

This module keeps the original public function names used by the existing
pipeline, but makes the reductions safer and more stable:

- confidence-weighted keypoint loss divides by valid confidence mass instead of
  all joints, so low-confidence or missing joints do not dilute the loss;
- silhouette loss supports robust Smooth-L1 or MSE behavior and normalizes by
  active visibility/region weight mass;
- pseudo landmark loss always returns a Tensor on the correct device, not a
  Python float, so downstream autograd/device code stays stable;
- priors use robust defaults and optional weighting;
- smoothness/laplacian losses are safer for empty inputs and batched tensors.

The old calls still work:
    keypoint_loss(pred_keypoints, gt_keypoints, confidence)
    silhouette_loss(pred_mask, target_mask, visibility_mask, region_weights)
    pseudo_landmark_loss(pred_value, target_value, weight=1.0)
    shape_prior_loss(betas)
    pose_prior_loss(body_pose)
    translation_loss(transl)
    smoothness_loss(vertices)
    laplacian_loss(offsets, neighbors)
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F

TensorLike = Union[torch.Tensor, float, int]


# =============================================================================
# Small utilities
# =============================================================================

def _as_tensor_like(value: TensorLike, ref: torch.Tensor) -> torch.Tensor:
    """Return value as a tensor on ref's device/dtype."""
    if torch.is_tensor(value):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(value, device=ref.device, dtype=ref.dtype)


def _zero_like_reference(*refs: Optional[torch.Tensor]) -> torch.Tensor:
    """Return scalar zero on the first available tensor's device/dtype."""
    for ref in refs:
        if torch.is_tensor(ref):
            return torch.zeros((), device=ref.device, dtype=ref.dtype)
    return torch.zeros((), dtype=torch.float32)


def _safe_weighted_mean(loss: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Weighted mean that normalizes by the active weight mass.

    This is better than plain `.mean()` when visibility/confidence masks contain
    many zeros. If all weights are zero, it returns a differentiable scalar zero.
    """
    if weight is None:
        if loss.numel() == 0:
            return torch.zeros((), device=loss.device, dtype=loss.dtype)
        return loss.mean()

    weight = weight.to(device=loss.device, dtype=loss.dtype)
    loss, weight = torch.broadcast_tensors(loss, weight)
    denom = weight.sum().clamp_min(eps)
    return (loss * weight).sum() / denom


def _elementwise_robust_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = "smooth_l1",
    beta: float = 0.01,
) -> torch.Tensor:
    """Elementwise robust loss with no reduction."""
    kind = str(kind).lower()
    if kind in {"l2", "mse", "squared"}:
        return (pred - target).square()
    if kind in {"l1", "abs", "mae"}:
        return (pred - target).abs()
    if kind in {"smooth_l1", "huber"}:
        # PyTorch Smooth-L1 uses a squared term below beta and an L1 term above it.
        return F.smooth_l1_loss(pred, target, reduction="none", beta=float(beta))
    raise ValueError(f"Unknown robust loss kind: {kind!r}")


def _reduce_last_dim(loss: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Reduce coordinate/channel loss over the final dimension."""
    if loss.dim() == 0:
        return loss
    mode = str(mode).lower()
    if mode == "sum":
        return loss.sum(dim=-1)
    if mode == "mean":
        return loss.mean(dim=-1)
    if mode == "none":
        return loss
    raise ValueError(f"Unsupported final-dim reduction: {mode!r}")


# =============================================================================
# KEYPOINT LOSS
# =============================================================================

def keypoint_loss(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    confidence: torch.Tensor,
    confidence_threshold: float = 0.4,
    loss_type: str = "smooth_l1",
    beta: float = 8.0,
    normalize_by_confidence: bool = True,
) -> torch.Tensor:
    """
    Confidence-weighted 2D/3D keypoint loss.

    Args:
        pred_keypoints: Tensor [..., K, C].
        gt_keypoints: Tensor broadcastable to pred_keypoints.
        confidence: Tensor [..., K] or [..., K, 1]. Values below threshold are ignored.
        confidence_threshold: Minimum keypoint confidence to contribute.
        loss_type: "smooth_l1", "l1", or "mse".
        beta: Smooth-L1 beta in the same unit as keypoints. If keypoints are in
            pixels, 4--12 is usually reasonable; if normalized, use a smaller beta.
        normalize_by_confidence: If True, divide by active confidence mass. This
            avoids diluting the loss when many keypoints are missing.
    """
    if pred_keypoints is None or gt_keypoints is None or confidence is None:
        return _zero_like_reference(pred_keypoints, gt_keypoints, confidence)

    gt_keypoints = gt_keypoints.to(device=pred_keypoints.device, dtype=pred_keypoints.dtype)
    confidence = confidence.to(device=pred_keypoints.device, dtype=pred_keypoints.dtype)

    if confidence.dim() == pred_keypoints.dim() and confidence.shape[-1] == 1:
        confidence = confidence.squeeze(-1)

    active = torch.where(
        confidence > float(confidence_threshold),
        confidence,
        torch.zeros_like(confidence),
    )

    coord_loss = _elementwise_robust_loss(pred_keypoints, gt_keypoints, kind=loss_type, beta=beta)
    per_joint = _reduce_last_dim(coord_loss, mode="sum")

    if normalize_by_confidence:
        return _safe_weighted_mean(per_joint, active)
    return (per_joint * active).mean()


# =============================================================================
# SILHOUETTE LOSS
# =============================================================================

def silhouette_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    visibility_mask: Optional[torch.Tensor],
    region_weights: Optional[torch.Tensor],
    loss_type: str = "mse",
    beta: float = 0.05,
    normalize_by_weight: bool = True,
    fp_weight: float = 1.0,
    fn_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted silhouette mismatch loss.

    Keeps the original API but fixes normalization. Optionally weighs false
    positives and false negatives differently:
      - false positive: pred > target, rendered body outside target mask;
      - false negative: target > pred, missing target coverage.
    """
    if pred_mask is None or target_mask is None:
        return _zero_like_reference(pred_mask, target_mask, visibility_mask, region_weights)

    target_mask = target_mask.to(device=pred_mask.device, dtype=pred_mask.dtype)

    weight = torch.ones_like(pred_mask)
    if visibility_mask is not None:
        weight = weight * visibility_mask.to(device=pred_mask.device, dtype=pred_mask.dtype)
    if region_weights is not None:
        rw = region_weights.to(device=pred_mask.device, dtype=pred_mask.dtype)
        # Accept [1,H,W], [1,1,H,W], [B,H,W], or [B,1,H,W].
        while rw.dim() < pred_mask.dim():
            rw = rw.unsqueeze(0)
        weight = weight * rw

    base = _elementwise_robust_loss(pred_mask, target_mask, kind=loss_type, beta=beta)

    if fp_weight != 1.0 or fn_weight != 1.0:
        diff = pred_mask - target_mask
        asym = torch.ones_like(base)
        asym = torch.where(diff > 0, asym * float(fp_weight), asym)
        asym = torch.where(diff < 0, asym * float(fn_weight), asym)
        base = base * asym

    if normalize_by_weight:
        return _safe_weighted_mean(base, weight)
    return (base * weight).mean()


def silhouette_fp_fn_loss(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    visibility_mask: Optional[torch.Tensor] = None,
    fp_region_weights: Optional[torch.Tensor] = None,
    fn_region_weights: Optional[torch.Tensor] = None,
    fp_weight: float = 1.0,
    fn_weight: float = 1.0,
    beta: float = 0.05,
) -> torch.Tensor:
    """
    Separate false-positive and false-negative silhouette penalties.

    This is useful for body fitting because bloat and missing coverage often
    need different regional weighting.
    """
    if pred_mask is None or target_mask is None:
        return _zero_like_reference(pred_mask, target_mask, visibility_mask, fp_region_weights, fn_region_weights)

    target_mask = target_mask.to(device=pred_mask.device, dtype=pred_mask.dtype)
    vis = torch.ones_like(pred_mask) if visibility_mask is None else visibility_mask.to(device=pred_mask.device, dtype=pred_mask.dtype)

    fp = torch.relu(pred_mask - target_mask)
    fn = torch.relu(target_mask - pred_mask)

    fp_loss = F.smooth_l1_loss(fp, torch.zeros_like(fp), reduction="none", beta=float(beta))
    fn_loss = F.smooth_l1_loss(fn, torch.zeros_like(fn), reduction="none", beta=float(beta))

    fp_w = vis
    fn_w = vis
    if fp_region_weights is not None:
        rw = fp_region_weights.to(device=pred_mask.device, dtype=pred_mask.dtype)
        while rw.dim() < pred_mask.dim():
            rw = rw.unsqueeze(0)
        fp_w = fp_w * rw
    if fn_region_weights is not None:
        rw = fn_region_weights.to(device=pred_mask.device, dtype=pred_mask.dtype)
        while rw.dim() < pred_mask.dim():
            rw = rw.unsqueeze(0)
        fn_w = fn_w * rw

    return float(fp_weight) * _safe_weighted_mean(fp_loss, fp_w) + float(fn_weight) * _safe_weighted_mean(fn_loss, fn_w)


# =============================================================================
# PSEUDO LANDMARK LOSS
# =============================================================================

def pseudo_landmark_loss(
    pred_value: Optional[torch.Tensor],
    target_value: Optional[torch.Tensor],
    weight: float = 1.0,
    confidence: Optional[torch.Tensor] = None,
    loss_type: str = "smooth_l1",
    beta: float = 4.0,
) -> torch.Tensor:
    """
    Robust pseudo-landmark loss.

    Unlike the old implementation, missing values return a scalar tensor rather
    than Python 0.0. That keeps device/autograd behavior predictable.
    """
    if pred_value is None or target_value is None:
        return _zero_like_reference(pred_value, target_value, confidence)

    target_value = target_value.to(device=pred_value.device, dtype=pred_value.dtype)
    elem = _elementwise_robust_loss(pred_value, target_value, kind=loss_type, beta=beta)

    if confidence is not None:
        conf = confidence.to(device=pred_value.device, dtype=pred_value.dtype)
        while conf.dim() < elem.dim():
            conf = conf.unsqueeze(-1)
        loss = _safe_weighted_mean(elem, conf)
    else:
        loss = elem.mean()

    return loss * float(weight)


def point_set_loss(
    pred_points: Optional[torch.Tensor],
    target_points: Optional[torch.Tensor],
    confidence: Optional[torch.Tensor] = None,
    weight: float = 1.0,
    loss_type: str = "smooth_l1",
    beta: float = 4.0,
) -> torch.Tensor:
    """Convenience loss for projected curves/point sets [..., N, C]."""
    if pred_points is None or target_points is None:
        return _zero_like_reference(pred_points, target_points, confidence)
    target_points = target_points.to(device=pred_points.device, dtype=pred_points.dtype)
    elem = _elementwise_robust_loss(pred_points, target_points, kind=loss_type, beta=beta)
    per_point = _reduce_last_dim(elem, mode="sum")
    if confidence is not None:
        return float(weight) * _safe_weighted_mean(per_point, confidence.to(device=pred_points.device, dtype=pred_points.dtype))
    return float(weight) * per_point.mean()


# =============================================================================
# SHAPE / POSE / TRANSLATION PRIORS
# =============================================================================

def shape_prior_loss(
    betas: Optional[torch.Tensor],
    beta_weights: Optional[torch.Tensor] = None,
    robust: bool = False,
    beta: float = 1.0,
) -> torch.Tensor:
    if betas is None:
        return _zero_like_reference(betas)
    if beta_weights is not None:
        w = beta_weights.to(device=betas.device, dtype=betas.dtype)
        while w.dim() < betas.dim():
            w = w.unsqueeze(0)
    else:
        w = None
    if robust:
        loss = F.smooth_l1_loss(betas, torch.zeros_like(betas), reduction="none", beta=float(beta))
    else:
        loss = betas.square()
    return _safe_weighted_mean(loss, w)


def pose_prior_loss(
    body_pose: Optional[torch.Tensor],
    pose_weights: Optional[torch.Tensor] = None,
    robust: bool = True,
    beta: float = 0.25,
) -> torch.Tensor:
    if body_pose is None:
        return _zero_like_reference(body_pose)
    if pose_weights is not None:
        w = pose_weights.to(device=body_pose.device, dtype=body_pose.dtype)
        while w.dim() < body_pose.dim():
            w = w.unsqueeze(0)
    else:
        w = None
    if robust:
        loss = F.smooth_l1_loss(body_pose, torch.zeros_like(body_pose), reduction="none", beta=float(beta))
    else:
        loss = body_pose.square()
    return _safe_weighted_mean(loss, w)


def translation_loss(
    transl: Optional[torch.Tensor],
    target: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if transl is None:
        return _zero_like_reference(transl, target, weights)
    if target is None:
        target = torch.zeros_like(transl)
    else:
        target = target.to(device=transl.device, dtype=transl.dtype)
    loss = (transl - target).square()
    return _safe_weighted_mean(loss, weights)


# =============================================================================
# SMOOTHNESS / LAPLACIAN REGULARIZATION
# =============================================================================

def smoothness_loss(
    vertices: Optional[torch.Tensor],
    dim: int = 1,
    weights: Optional[torch.Tensor] = None,
    loss_type: str = "mse",
    beta: float = 0.01,
) -> torch.Tensor:
    """
    Simple adjacent-index smoothness.

    For actual mesh smoothness, prefer laplacian_loss with mesh neighbors. This
    function remains for compatibility with older code paths.
    """
    if vertices is None or vertices.numel() == 0:
        return _zero_like_reference(vertices, weights)
    if vertices.shape[dim] < 2:
        return _zero_like_reference(vertices, weights)

    a = vertices.narrow(dim, 1, vertices.shape[dim] - 1)
    b = vertices.narrow(dim, 0, vertices.shape[dim] - 1)
    diff = a - b
    if loss_type.lower() in {"smooth_l1", "huber"}:
        loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=float(beta))
    elif loss_type.lower() in {"l1", "abs", "mae"}:
        loss = diff.abs()
    else:
        loss = diff.square()
    return _safe_weighted_mean(loss, weights)


def laplacian_loss(
    offsets: Optional[torch.Tensor],
    neighbors: Optional[Mapping[int, Sequence[int]]],
    vertex_weights: Optional[torch.Tensor] = None,
    loss_type: str = "mse",
    beta: float = 0.01,
) -> torch.Tensor:
    """
    Mesh Laplacian regularization for offset/deformation tensors.

    Args:
        offsets: [V,C] or [B,V,C].
        neighbors: dict vertex_id -> sequence of neighbor vertex ids.
        vertex_weights: optional [V] or [B,V] weights.
    """
    if offsets is None or neighbors is None or len(neighbors) == 0:
        return _zero_like_reference(offsets, vertex_weights)

    was_unbatched = False
    if offsets.dim() == 2:
        offsets = offsets.unsqueeze(0)
        was_unbatched = True
    if offsets.dim() != 3:
        raise ValueError(f"Expected offsets with shape [V,C] or [B,V,C], got {tuple(offsets.shape)}")

    bsz, vcount, _ = offsets.shape
    terms = []
    weights = []

    for vid, nbs in neighbors.items():
        if vid < 0 or vid >= vcount or not nbs:
            continue
        valid_nbs = [int(n) for n in nbs if 0 <= int(n) < vcount]
        if not valid_nbs:
            continue
        v = offsets[:, int(vid), :]
        nb = offsets[:, valid_nbs, :].mean(dim=1)
        diff = v - nb
        if loss_type.lower() in {"smooth_l1", "huber"}:
            term = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=float(beta)).mean(dim=-1)
        elif loss_type.lower() in {"l1", "abs", "mae"}:
            term = diff.abs().mean(dim=-1)
        else:
            term = diff.square().mean(dim=-1)
        terms.append(term)

        if vertex_weights is not None:
            vw = vertex_weights.to(device=offsets.device, dtype=offsets.dtype)
            if vw.dim() == 1:
                weights.append(vw[int(vid)].expand(bsz))
            elif vw.dim() == 2:
                weights.append(vw[:, int(vid)])
            else:
                raise ValueError("vertex_weights must have shape [V] or [B,V]")

    if not terms:
        return _zero_like_reference(offsets, vertex_weights)

    loss = torch.stack(terms, dim=1)  # [B, N]
    if vertex_weights is not None:
        w = torch.stack(weights, dim=1)
        return _safe_weighted_mean(loss, w)
    return loss.mean()


# =============================================================================
# Extra optional helpers for newer optimizer code paths
# =============================================================================

def deformation_magnitude_loss(
    offsets: Optional[torch.Tensor],
    vertex_weights: Optional[torch.Tensor] = None,
    p: int = 2,
) -> torch.Tensor:
    """Penalize per-vertex deformation magnitude."""
    if offsets is None:
        return _zero_like_reference(offsets, vertex_weights)
    mag = offsets.norm(p=p, dim=-1)
    return _safe_weighted_mean(mag.square(), vertex_weights)


def edge_length_preservation_loss(
    vertices: Optional[torch.Tensor],
    reference_vertices: Optional[torch.Tensor],
    edges: Optional[torch.Tensor],
    edge_weights: Optional[torch.Tensor] = None,
    loss_type: str = "smooth_l1",
    beta: float = 0.005,
) -> torch.Tensor:
    """Preserve edge lengths relative to a reference mesh."""
    if vertices is None or reference_vertices is None or edges is None or edges.numel() == 0:
        return _zero_like_reference(vertices, reference_vertices, edges, edge_weights)
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    if reference_vertices.dim() == 2:
        reference_vertices = reference_vertices.unsqueeze(0)
    edges = edges.to(device=vertices.device, dtype=torch.long)
    a, b = edges[:, 0], edges[:, 1]
    cur = (vertices[:, a, :] - vertices[:, b, :]).norm(dim=-1)
    ref = (reference_vertices[:, a, :] - reference_vertices[:, b, :]).norm(dim=-1)
    elem = _elementwise_robust_loss(cur, ref, kind=loss_type, beta=beta)
    return _safe_weighted_mean(elem, edge_weights)
