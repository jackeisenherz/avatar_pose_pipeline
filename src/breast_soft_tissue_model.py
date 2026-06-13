"""
Differentiable breast soft-tissue corrective layer for SMPL-X vertices.

This module keeps the SMPL-X model and topology unchanged. It takes the vertices
produced by SMPL-X and applies a small, controlled, differentiable deformation
restricted by the generated breast topology prior.

Typical use:

    soft = BreastSoftTissueModel(
        prior_json='assets/smplx_female_breast_topology_prior.json',
        weights_npz='assets/smplx_female_breast_soft_weights.npz',
        device=device,
    )
    smplx_out = smplx_model(...)
    corrected_vertices = soft(smplx_out.vertices, smplx_out.joints)

Parameters are in meters if your SMPL-X vertices are in meters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


class BreastSoftTissueModel(nn.Module):
    def __init__(
        self,
        prior_json: str | Path,
        weights_npz: str | Path,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        max_projection_m: float = 0.13,
        max_sag_m: float = 0.16,
        max_lateral_spread_m: float = 0.08,
        max_fold_depth_m: float = 0.045,
        max_lower_fullness_m: float = 0.08,
    ):
        super().__init__()
        self.prior_json = Path(prior_json)
        self.weights_npz = Path(weights_npz)
        self.device_arg = device
        self.dtype_arg = dtype

        prior = json.loads(self.prior_json.read_text())
        self.prior = prior
        self.vertex_count = int(prior["vertex_count"])
        coord = prior.get("coordinate_convention", {})
        self.depth_axis = int(coord.get("template_depth_axis", 2))
        self.front_sign = float(coord.get("template_front_sign", 1.0))

        w = np.load(str(self.weights_npz))
        required = [
            "left_breast", "right_breast", "left_imf", "right_imf",
            "left_lower_pole", "right_lower_pole", "sternum",
            "left_armpit_guard", "right_armpit_guard", "upper_chest_guard", "abdomen_guard",
        ]
        for key in required:
            if key not in w:
                raise KeyError(f"weights file missing {key}")
            arr = torch.as_tensor(np.asarray(w[key]), dtype=dtype, device=device)
            if arr.numel() != self.vertex_count:
                raise ValueError(f"{key} has {arr.numel()} weights, expected {self.vertex_count}")
            self.register_buffer(f"w_{key}", arr.reshape(1, -1, 1), persistent=False)

        # Scale constants. Unconstrained parameters are mapped with tanh to safe ranges.
        self.max_projection_m = float(max_projection_m)
        self.max_sag_m = float(max_sag_m)
        self.max_lateral_spread_m = float(max_lateral_spread_m)
        self.max_fold_depth_m = float(max_fold_depth_m)
        self.max_lower_fullness_m = float(max_lower_fullness_m)

        # Left/right parameters. Initialize at zero: no correction.
        self.left_projection_raw = nn.Parameter(torch.zeros(()))
        self.right_projection_raw = nn.Parameter(torch.zeros(()))
        self.left_sag_raw = nn.Parameter(torch.zeros(()))
        self.right_sag_raw = nn.Parameter(torch.zeros(()))
        self.left_lateral_spread_raw = nn.Parameter(torch.zeros(()))
        self.right_lateral_spread_raw = nn.Parameter(torch.zeros(()))
        self.left_fold_depth_raw = nn.Parameter(torch.zeros(()))
        self.right_fold_depth_raw = nn.Parameter(torch.zeros(()))
        self.left_lower_fullness_raw = nn.Parameter(torch.zeros(()))
        self.right_lower_fullness_raw = nn.Parameter(torch.zeros(()))

    def parameters_as_dict(self) -> Dict[str, float]:
        with torch.no_grad():
            return {
                "left_projection_m": float(torch.tanh(self.left_projection_raw) * self.max_projection_m),
                "right_projection_m": float(torch.tanh(self.right_projection_raw) * self.max_projection_m),
                "left_sag_m": float(torch.tanh(self.left_sag_raw) * self.max_sag_m),
                "right_sag_m": float(torch.tanh(self.right_sag_raw) * self.max_sag_m),
                "left_lateral_spread_m": float(torch.tanh(self.left_lateral_spread_raw) * self.max_lateral_spread_m),
                "right_lateral_spread_m": float(torch.tanh(self.right_lateral_spread_raw) * self.max_lateral_spread_m),
                "left_fold_depth_m": float(torch.tanh(self.left_fold_depth_raw) * self.max_fold_depth_m),
                "right_fold_depth_m": float(torch.tanh(self.right_fold_depth_raw) * self.max_fold_depth_m),
                "left_lower_fullness_m": float(torch.tanh(self.left_lower_fullness_raw) * self.max_lower_fullness_m),
                "right_lower_fullness_m": float(torch.tanh(self.right_lower_fullness_raw) * self.max_lower_fullness_m),
            }

    def _compute_frame(self, vertices: torch.Tensor, joints: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute a stable chest frame. Returns origin, right, up, front for each batch.
        If joints are available and in SMPL-X joint order with shoulders/hips, use them.
        Otherwise use template coordinate axes in the current vertex space.
        """
        B = vertices.shape[0]
        dev = vertices.device
        dtype = vertices.dtype

        if joints is not None and joints.ndim == 3 and joints.shape[1] > 17:
            # Common SMPL-X body joint order: shoulders 16/17, hips 1/2. If your wrapper
            # uses another order, pass corrected joints or replace this block.
            left_hip = joints[:, 1]
            right_hip = joints[:, 2]
            left_shoulder = joints[:, 16]
            right_shoulder = joints[:, 17]
            origin = 0.5 * (left_shoulder + right_shoulder)
            right_axis = _normalize(left_shoulder - right_shoulder)
            up_axis = _normalize(0.5 * (left_shoulder + right_shoulder) - 0.5 * (left_hip + right_hip))
            front_axis = _normalize(torch.cross(right_axis, up_axis, dim=-1))
            # Ensure right/up/front are orthonormal.
            up_axis = _normalize(torch.cross(front_axis, right_axis, dim=-1))
            return origin, right_axis, up_axis, front_axis

        # Fallback for template-like coordinates.
        origin = vertices.mean(dim=1)
        right_axis = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dtype).reshape(1, 3).expand(B, 3)
        up_axis = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dtype).reshape(1, 3).expand(B, 3)
        front_vec = [0.0, 0.0, 0.0]
        front_vec[self.depth_axis] = self.front_sign
        front_axis = torch.tensor(front_vec, device=dev, dtype=dtype).reshape(1, 3).expand(B, 3)
        return origin, right_axis, up_axis, front_axis

    def forward(
        self,
        vertices: torch.Tensor,
        joints: Optional[torch.Tensor] = None,
        gravity_world: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vertices: [B,V,3] SMPL-X output vertices.
            joints: optional [B,J,3] joints for chest-frame estimation.
            gravity_world: optional [3] or [B,3] gravity direction in world coordinates.

        Returns:
            corrected vertices [B,V,3].
        """
        if vertices.ndim != 3 or vertices.shape[1] != self.vertex_count or vertices.shape[2] != 3:
            raise ValueError(f"vertices must be [B,{self.vertex_count},3], got {tuple(vertices.shape)}")

        B = vertices.shape[0]
        origin, right, up, front = self._compute_frame(vertices, joints)

        if gravity_world is None:
            # Default: world down. If your fitted scene uses different coordinates,
            # pass gravity_world explicitly.
            gravity = -up
        else:
            gravity = gravity_world.to(device=vertices.device, dtype=vertices.dtype)
            if gravity.ndim == 1:
                gravity = gravity.reshape(1, 3).expand(B, 3)
            gravity = _normalize(gravity)

        # Scalars mapped to safe ranges.
        lp = torch.tanh(self.left_projection_raw) * self.max_projection_m
        rp = torch.tanh(self.right_projection_raw) * self.max_projection_m
        ls = torch.tanh(self.left_sag_raw) * self.max_sag_m
        rs = torch.tanh(self.right_sag_raw) * self.max_sag_m
        ll = torch.tanh(self.left_lateral_spread_raw) * self.max_lateral_spread_m
        rl = torch.tanh(self.right_lateral_spread_raw) * self.max_lateral_spread_m
        lf = torch.tanh(self.left_fold_depth_raw) * self.max_fold_depth_m
        rf = torch.tanh(self.right_fold_depth_raw) * self.max_fold_depth_m
        llower = torch.tanh(self.left_lower_fullness_raw) * self.max_lower_fullness_m
        rlower = torch.tanh(self.right_lower_fullness_raw) * self.max_lower_fullness_m

        front_b = front.reshape(B, 1, 3)
        right_b = right.reshape(B, 1, 3)
        gravity_b = gravity.reshape(B, 1, 3)

        disp = torch.zeros_like(vertices)

        # Projection / size outward.
        disp = disp + self.w_left_breast * lp * front_b
        disp = disp + self.w_right_breast * rp * front_b

        # Lower pole fullness: outward and slightly down.
        disp = disp + self.w_left_lower_pole * llower * (0.70 * front_b + 0.30 * gravity_b)
        disp = disp + self.w_right_lower_pole * rlower * (0.70 * front_b + 0.30 * gravity_b)

        # Sag: mostly gravity direction, weighted by lower pole and whole breast.
        disp = disp + self.w_left_breast * ls * gravity_b
        disp = disp + self.w_right_breast * rs * gravity_b

        # Lateral spread: left goes +right_axis, right goes -right_axis by convention.
        disp = disp + self.w_left_breast * ll * right_b
        disp = disp - self.w_right_breast * rl * right_b

        # IMF crease: inward indentation around fold band. Add small anchoring upward/downward via gravity.
        disp = disp - self.w_left_imf * lf * front_b + self.w_left_imf * (0.18 * lf) * (-gravity_b)
        disp = disp - self.w_right_imf * rf * front_b + self.w_right_imf * (0.18 * rf) * (-gravity_b)

        # Guard regularization is returned via regularization_loss, not applied as hard clipping.
        return vertices + disp

    def regularization_loss(self) -> torch.Tensor:
        """Small prior to keep deformations plausible and avoid runaway values."""
        params = [
            self.left_projection_raw, self.right_projection_raw,
            self.left_sag_raw, self.right_sag_raw,
            self.left_lateral_spread_raw, self.right_lateral_spread_raw,
            self.left_fold_depth_raw, self.right_fold_depth_raw,
            self.left_lower_fullness_raw, self.right_lower_fullness_raw,
        ]
        return sum((p ** 2 for p in params)) * 0.01

    def guard_deformation_loss(self, base_vertices: torch.Tensor, corrected_vertices: torch.Tensor) -> torch.Tensor:
        """Penalize deformation leakage into sternum/armpit/upper-chest/abdomen guard regions."""
        d = torch.linalg.norm(corrected_vertices - base_vertices, dim=-1, keepdim=True)
        guard = (
            1.00 * self.w_sternum
            + 0.85 * self.w_left_armpit_guard
            + 0.85 * self.w_right_armpit_guard
            + 0.60 * self.w_upper_chest_guard
            + 0.75 * self.w_abdomen_guard
        ).clamp(0.0, 1.0)
        return (guard * d.square()).mean()
