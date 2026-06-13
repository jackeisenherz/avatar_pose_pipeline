# Drop-in replacement for: src/refinement/laplacian.py
"""
Mesh smoothness utilities for refinement.

This implementation keeps the original public API:

    reg = LaplacianRegularizer(faces, device="cuda")
    reg.offset_loss(offsets)
    reg.surface_loss(vertices)

and adds safer, more useful losses for normal-offset based refinement:

    reg.edge_smoothness(values)
    reg.uniform_laplacian_loss(values)
    reg.edge_length_loss(vertices, reference_vertices)
    reg.normal_consistency_loss(vertices)
    reg.offset_magnitude_loss(offsets)

Notes
-----
For freeform refinement, regularizing *offsets* is usually safer than
regularizing absolute vertices, because it preserves the fitted SMPL-X surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LaplacianStats:
    num_vertices: int
    num_faces: int
    num_edges: int


class LaplacianRegularizer:
    """
    Edge and uniform-Laplacian regularizer for triangular meshes.

    Parameters
    ----------
    faces:
        Tensor/array of shape [F, 3] or [1, F, 3].
    num_vertices:
        Optional vertex count. If omitted, inferred from faces.
    device:
        Target device.
    dtype:
        Floating dtype used for degree buffers.

    Important behavior
    ------------------
    - `offset_loss(offsets)` regularizes local offset differences.
    - `surface_loss(vertices)` regularizes absolute surface edge differences.
      This is provided for compatibility, but for refinement you usually want
      `offset_loss` and/or `uniform_laplacian_loss(offsets)`.
    """

    def __init__(
        self,
        faces,
        num_vertices: Optional[int] = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        if isinstance(faces, torch.Tensor):
            f = faces.detach().long().cpu()
        else:
            f = torch.as_tensor(faces, dtype=torch.long)

        while f.dim() > 2:
            f = f[0]

        if f.dim() != 2 or f.shape[-1] != 3:
            raise ValueError(f"faces must have shape [F,3] or [1,F,3], got {tuple(f.shape)}")

        self.faces = f.to(device=self.device, dtype=torch.long)
        self.num_faces = int(self.faces.shape[0])
        self.num_vertices = int(num_vertices) if num_vertices is not None else int(self.faces.max().item() + 1)

        edges = torch.cat(
            [
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            ],
            dim=0,
        )
        edges = torch.sort(edges, dim=1).values
        edges = torch.unique(edges, dim=0)
        self.edges = edges.to(device=self.device, dtype=torch.long)
        self.num_edges = int(self.edges.shape[0])

        # Directed edge list for neighbor aggregation.
        src = torch.cat([self.edges[:, 0], self.edges[:, 1]], dim=0)
        dst = torch.cat([self.edges[:, 1], self.edges[:, 0]], dim=0)
        self._src = src
        self._dst = dst

        degree = torch.zeros(self.num_vertices, 1, device=self.device, dtype=self.dtype)
        degree.index_add_(0, self._src, torch.ones_like(self._src, dtype=self.dtype, device=self.device).unsqueeze(1))
        self.degree = degree.clamp_min(1.0)

        self.stats = LaplacianStats(
            num_vertices=self.num_vertices,
            num_faces=self.num_faces,
            num_edges=self.num_edges,
        )

    # ---------------------------------------------------------------------
    # Shape helpers
    # ---------------------------------------------------------------------

    def _batched(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() == 2:
            values = values.unsqueeze(0)
        if values.dim() != 3:
            raise ValueError(f"expected values [V,C] or [B,V,C], got {tuple(values.shape)}")
        if values.shape[1] < self.num_vertices:
            raise ValueError(
                f"values has {values.shape[1]} vertices, but faces require at least {self.num_vertices}"
            )
        return values

    def neighbor_mean(self, values: torch.Tensor) -> torch.Tensor:
        """
        Mean of 1-ring neighbor values.

        values: [B,V,C] or [V,C]
        returns same rank as input.
        """
        input_was_unbatched = values.dim() == 2
        values_b = self._batched(values)

        outs = []
        for b in range(values_b.shape[0]):
            base = values_b[b]
            sums = torch.zeros_like(base)
            sums.index_add_(0, self._src, base[self._dst])
            outs.append(sums / self.degree.to(dtype=base.dtype))

        out = torch.stack(outs, dim=0)
        return out[0] if input_was_unbatched else out

    # ---------------------------------------------------------------------
    # Losses
    # ---------------------------------------------------------------------

    def edge_smoothness(self, values: torch.Tensor, robust: bool = False, eps: float = 1e-6) -> torch.Tensor:
        """
        Penalize differences across mesh edges.

        For scalar normal offsets, use values [B,V,1].
        For XYZ offsets/vertices, use values [B,V,3].
        """
        values = self._batched(values)
        a = values[:, self.edges[:, 0]]
        b = values[:, self.edges[:, 1]]
        diff2 = (a - b).square().sum(dim=-1)
        if robust:
            return torch.sqrt(diff2 + eps).mean()
        return diff2.mean()

    def uniform_laplacian(self, values: torch.Tensor) -> torch.Tensor:
        """
        Uniform graph Laplacian: value - mean(neighbor values).
        """
        values = self._batched(values)
        return values - self.neighbor_mean(values)

    def uniform_laplacian_loss(self, values: torch.Tensor, robust: bool = True, eps: float = 1e-6) -> torch.Tensor:
        lap = self.uniform_laplacian(values)
        diff2 = lap.square().sum(dim=-1)
        if robust:
            return torch.sqrt(diff2 + eps).mean()
        return diff2.mean()

    def offset_loss(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible offset smoothness loss.
        """
        return self.edge_smoothness(offsets)

    def surface_loss(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible absolute surface smoothness loss.

        Usually less safe than `offset_loss` for body refinement because it can
        shrink or distort the surface if overweighted.
        """
        return self.edge_smoothness(vertices)

    def offset_magnitude_loss(self, offsets: torch.Tensor, robust: bool = False, eps: float = 1e-6) -> torch.Tensor:
        offsets = self._batched(offsets)
        mag2 = offsets.square().sum(dim=-1)
        if robust:
            return torch.sqrt(mag2 + eps).mean()
        return mag2.mean()

    def edge_length_loss(
        self,
        vertices: torch.Tensor,
        reference_vertices: torch.Tensor,
        relative: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Penalize changes in edge lengths relative to a reference mesh.
        """
        vertices = self._batched(vertices)
        reference_vertices = self._batched(reference_vertices)

        if reference_vertices.shape[0] == 1 and vertices.shape[0] > 1:
            reference_vertices = reference_vertices.expand(vertices.shape[0], -1, -1)

        v0 = vertices[:, self.edges[:, 0]]
        v1 = vertices[:, self.edges[:, 1]]
        r0 = reference_vertices[:, self.edges[:, 0]]
        r1 = reference_vertices[:, self.edges[:, 1]]

        el = torch.linalg.norm(v0 - v1, dim=-1)
        rl = torch.linalg.norm(r0 - r1, dim=-1).clamp_min(eps)

        if relative:
            return ((el - rl) / rl).square().mean()
        return (el - rl).square().mean()

    def vertex_normals(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Area-weighted vertex normals.
        """
        input_was_unbatched = vertices.dim() == 2
        vertices = self._batched(vertices)

        f = self.faces
        outs = []
        for b in range(vertices.shape[0]):
            v = vertices[b]
            v0 = v[f[:, 0]]
            v1 = v[f[:, 1]]
            v2 = v[f[:, 2]]
            face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

            normals = torch.zeros_like(v)
            normals.index_add_(0, f[:, 0], face_normals)
            normals.index_add_(0, f[:, 1], face_normals)
            normals.index_add_(0, f[:, 2], face_normals)
            outs.append(F.normalize(normals, dim=-1, eps=1e-8))

        out = torch.stack(outs, dim=0)
        return out[0] if input_was_unbatched else out

    def normal_consistency_loss(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Penalize neighboring vertex normals that diverge.
        """
        n = self.vertex_normals(vertices)
        n = self._batched(n)
        a = n[:, self.edges[:, 0]]
        b = n[:, self.edges[:, 1]]
        return (1.0 - (a * b).sum(dim=-1).clamp(-1.0, 1.0)).mean()
