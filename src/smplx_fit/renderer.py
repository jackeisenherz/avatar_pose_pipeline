"""
renderer_improved.py

Safe differentiable silhouette renderer for SMPL-X fitting.

Drop-in replacement for src/smplx_fit/renderer.py.

Why this version exists:
    PyTorch3D can emit this warning during optimization:
        "Bin size was too small in the coarse rasterization phase..."

    For fitting/optimization, an incomplete silhouette is worse than a slower render.
    Therefore the default here is:
        bin_size=0

    In PyTorch3D this disables coarse binning and uses the naive rasterization path.
    It is slower, but robust and avoids incomplete rasterization caused by bin overflow.

Public API compatibility:
    SilhouetteRenderer(...).render(...) returns RGBA [B,H,W,4].
    The alpha/silhouette channel is rgba[..., 3].

Useful helpers:
    render_alpha(...)       -> [B,1,H,W]
    render_hard_mask(...)   -> [B,1,H,W]
    rasterize_fragments(...) for debugging visibility/z-buffer problems

Recommended use for your current optimization:

    renderer = SilhouetteRenderer(
        image_size=(height, width),
        device=device,
        faces_per_pixel=50,
        bin_size=0,   # safest; default
    )

If you later want speed and the warning is gone, try:

    renderer = SilhouetteRenderer(
        image_size=(height, width),
        device=device,
        faces_per_pixel=50,
        bin_size=32,
        max_faces_per_bin=200000,
    )
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftSilhouetteShader,
        BlendParams,
    )
except Exception as exc:  # pragma: no cover
    Meshes = None
    PerspectiveCameras = None
    RasterizationSettings = None
    MeshRenderer = None
    MeshRasterizer = None
    SoftSilhouetteShader = None
    BlendParams = None
    _PYTORCH3D_IMPORT_ERROR = exc
else:
    _PYTORCH3D_IMPORT_ERROR = None


ImageSize = Union[int, Tuple[int, int], Sequence[int]]
TensorLike = Union[float, int, Sequence[float], torch.Tensor]


class SilhouetteRenderer:
    """
    PyTorch3D soft-silhouette renderer.

    Coordinate convention:
        Screen-space intrinsics are used by default:
            in_ndc=False
            focal_length and principal_point are in pixels
            image_size is (height, width)

    Important rasterization defaults:
        bin_size=0 by default to avoid PyTorch3D coarse-rasterization bin overflow.
        This is the safe setting for optimization. It can be slower, but it prevents
        incomplete silhouette output.
    """

    def __init__(
        self,
        image_size: ImageSize = 1024,
        device: Union[str, torch.device] = "cuda",
        sigma: float = 1e-4,
        gamma: float = 1e-4,
        faces_per_pixel: int = 50,
        bin_size: Optional[int] = 0,
        max_faces_per_bin: Optional[int] = None,
        cull_backfaces: bool = False,
        perspective_correct: Optional[bool] = None,
        clip_barycentric_coords: Optional[bool] = None,
    ) -> None:
        self.device = torch.device(device)
        self.image_size = self._normalize_image_size(image_size)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.faces_per_pixel = int(faces_per_pixel)
        self.bin_size = bin_size
        self.max_faces_per_bin = max_faces_per_bin
        self.cull_backfaces = bool(cull_backfaces)
        self.perspective_correct = perspective_correct
        self.clip_barycentric_coords = clip_barycentric_coords

        self._check_pytorch3d_available()
        self._build_renderer()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _normalize_image_size(image_size: ImageSize) -> Tuple[int, int]:
        if isinstance(image_size, int):
            h = w = int(image_size)
        else:
            vals = list(image_size)
            if len(vals) != 2:
                raise ValueError(f"image_size must be int or (height, width), got {image_size!r}")
            h, w = int(vals[0]), int(vals[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size must be positive, got {(h, w)}")
        return h, w

    @property
    def height(self) -> int:
        return self.image_size[0]

    @property
    def width(self) -> int:
        return self.image_size[1]

    def _check_pytorch3d_available(self) -> None:
        if _PYTORCH3D_IMPORT_ERROR is not None:
            raise ImportError("PyTorch3D is required by SilhouetteRenderer but could not be imported.") from _PYTORCH3D_IMPORT_ERROR

    def _soft_blur_radius(self) -> float:
        """
        Recommended PyTorch3D relation for SoftSilhouetteShader:
            blur_radius = log(1 / sigma - 1) * sigma
        """
        sigma = min(max(float(self.sigma), 1e-8), 1.0 - 1e-8)
        return float(math.log(1.0 / sigma - 1.0) * sigma)

    def _build_renderer(self) -> None:
        self.blend_params = BlendParams(sigma=self.sigma, gamma=self.gamma)
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=self._soft_blur_radius(),
            faces_per_pixel=self.faces_per_pixel,
            # Critical fix for your warning:
            #   bin_size=0 disables coarse binning and avoids bin overflow.
            bin_size=self.bin_size,
            max_faces_per_bin=self.max_faces_per_bin,
            cull_backfaces=self.cull_backfaces,
            perspective_correct=self.perspective_correct,
            clip_barycentric_coords=self.clip_barycentric_coords,
        )
        self.rasterizer = MeshRasterizer(raster_settings=self.raster_settings)
        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=SoftSilhouetteShader(blend_params=self.blend_params),
        )

    # ------------------------------------------------------------------ config helpers
    def set_safe_rasterization(self) -> "SilhouetteRenderer":
        """Switch to robust naive rasterization and rebuild the renderer."""
        self.bin_size = 0
        self.max_faces_per_bin = None
        self._build_renderer()
        return self

    def set_fast_rasterization(self, bin_size: int = 32, max_faces_per_bin: int = 200000) -> "SilhouetteRenderer":
        """
        Switch to coarse rasterization with a large bin capacity.

        Use this only if the overflow warning no longer appears on your data.
        """
        self.bin_size = int(bin_size)
        self.max_faces_per_bin = int(max_faces_per_bin)
        self._build_renderer()
        return self

    def update_image_size(self, image_size: ImageSize) -> "SilhouetteRenderer":
        """Change image size and rebuild raster settings."""
        self.image_size = self._normalize_image_size(image_size)
        self._build_renderer()
        return self

    # ------------------------------------------------------------------ tensor helpers
    def _ensure_batched_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        if not isinstance(vertices, torch.Tensor):
            raise TypeError(f"vertices must be a torch.Tensor, got {type(vertices)!r}")
        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        if vertices.dim() != 3 or vertices.shape[-1] != 3:
            raise ValueError(f"vertices must have shape [V,3] or [B,V,3], got {tuple(vertices.shape)}")
        return vertices

    def _ensure_batched_faces(self, faces: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if not isinstance(faces, torch.Tensor):
            faces = torch.as_tensor(faces)
        faces = faces.to(device=device, dtype=torch.long)
        if faces.dim() == 2:
            faces = faces.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        elif faces.dim() == 3:
            if faces.shape[0] == 1 and batch_size > 1:
                faces = faces.expand(batch_size, -1, -1).contiguous()
            elif faces.shape[0] != batch_size:
                raise ValueError(f"faces batch size {faces.shape[0]} does not match vertices batch size {batch_size}")
        else:
            raise ValueError(f"faces must have shape [F,3] or [B,F,3], got {tuple(faces.shape)}")
        if faces.shape[-1] != 3:
            raise ValueError(f"faces last dimension must be 3, got {tuple(faces.shape)}")
        return faces

    def _batch_intrinsics(
        self,
        value: TensorLike,
        batch_size: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        t = torch.as_tensor(value, dtype=torch.float32, device=device)
        if t.dim() == 0:
            t = t.view(1, 1).expand(batch_size, 2).contiguous()
        elif t.dim() == 1:
            if t.numel() == 1:
                t = t.view(1, 1).expand(batch_size, 2).contiguous()
            elif t.numel() == 2:
                t = t.view(1, 2).expand(batch_size, 2).contiguous()
            elif t.numel() == batch_size:
                t = t.view(batch_size, 1).expand(batch_size, 2).contiguous()
            else:
                raise ValueError(f"{name} shape {tuple(t.shape)} cannot be broadcast to [B,2]")
        elif t.dim() == 2:
            if t.shape == (1, 2) and batch_size > 1:
                t = t.expand(batch_size, 2).contiguous()
            elif t.shape == (batch_size, 1):
                t = t.expand(batch_size, 2).contiguous()
            elif t.shape != (batch_size, 2):
                raise ValueError(f"{name} must have shape [B,2], [1,2], [B,1], or scalar, got {tuple(t.shape)}")
        else:
            raise ValueError(f"{name} must be scalar, [2], [B], [1,2], [B,1], or [B,2], got {tuple(t.shape)}")
        return t

    def _batch_translation(
        self,
        translation: Optional[Union[Sequence[float], torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if translation is None:
            t = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
            t[:, 2] = 2.5
            return t
        t = torch.as_tensor(translation, dtype=torch.float32, device=device)
        if t.dim() == 1:
            if t.numel() != 3:
                raise ValueError(f"translation must have 3 values or shape [B,3], got {tuple(t.shape)}")
            t = t.view(1, 3).expand(batch_size, 3).contiguous()
        elif t.dim() == 2:
            if t.shape == (1, 3) and batch_size > 1:
                t = t.expand(batch_size, 3).contiguous()
            elif t.shape != (batch_size, 3):
                raise ValueError(f"translation must have shape [B,3] or [1,3], got {tuple(t.shape)}")
        else:
            raise ValueError(f"translation must have shape [3], [1,3], or [B,3], got {tuple(t.shape)}")
        return t

    def _batch_rotation(
        self,
        rotation: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if rotation is None:
            return torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, 3, 3).contiguous()
        r = torch.as_tensor(rotation, dtype=torch.float32, device=device)
        if r.dim() == 2:
            if r.shape != (3, 3):
                raise ValueError(f"rotation must be [3,3] or [B,3,3], got {tuple(r.shape)}")
            r = r.unsqueeze(0).expand(batch_size, 3, 3).contiguous()
        elif r.dim() == 3:
            if r.shape == (1, 3, 3) and batch_size > 1:
                r = r.expand(batch_size, 3, 3).contiguous()
            elif r.shape != (batch_size, 3, 3):
                raise ValueError(f"rotation must be [B,3,3] or [1,3,3], got {tuple(r.shape)}")
        else:
            raise ValueError(f"rotation must be [3,3], [1,3,3], or [B,3,3], got {tuple(r.shape)}")
        return r

    def _image_size_tensor(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.tensor([list(self.image_size)], dtype=torch.float32, device=device).expand(batch_size, 2).contiguous()

    # ------------------------------------------------------------------ cameras / meshes
    def create_camera(
        self,
        focal_length: TensorLike = 1500.0,
        principal_point: Optional[TensorLike] = None,
        translation: Optional[Union[Sequence[float], torch.Tensor]] = None,
        rotation: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        device = torch.device(device) if device is not None else self.device
        if principal_point is None:
            principal_point = [self.width * 0.5, self.height * 0.5]
        focal = self._batch_intrinsics(focal_length, batch_size, device, "focal_length")
        center = self._batch_intrinsics(principal_point, batch_size, device, "principal_point")
        T = self._batch_translation(translation, batch_size, device)
        R = self._batch_rotation(rotation, batch_size, device)
        return PerspectiveCameras(
            focal_length=focal,
            principal_point=center,
            R=R,
            T=T,
            image_size=self._image_size_tensor(batch_size, device),
            in_ndc=False,
            device=device,
        )

    def create_meshes(self, vertices: torch.Tensor, faces: torch.Tensor) -> Meshes:
        vertices = self._ensure_batched_vertices(vertices)
        device = vertices.device
        faces = self._ensure_batched_faces(faces, int(vertices.shape[0]), device)
        return Meshes(verts=vertices, faces=faces)

    # ------------------------------------------------------------------ rendering
    def render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        focal_length: TensorLike = 1500.0,
        principal_point: Optional[TensorLike] = None,
        translation: Optional[Union[Sequence[float], torch.Tensor]] = None,
        cameras=None,
    ) -> torch.Tensor:
        """
        Render soft silhouettes.

        Args:
            vertices: [V,3] or [B,V,3]
            faces: [F,3] or [B,F,3]
            focal_length: scalar, [2], [B], [B,1], or [B,2] in pixels
            principal_point: [2], [1,2], or [B,2] in pixels. Defaults to image center.
            translation: optional [3], [1,3], or [B,3]
            cameras: optional pre-built PerspectiveCameras. If supplied,
                focal_length/principal_point/translation are ignored.

        Returns:
            RGBA tensor [B,H,W,4]. The alpha/silhouette channel is [..., 3].
        """
        vertices = self._ensure_batched_vertices(vertices)
        device = vertices.device
        batch_size = int(vertices.shape[0])
        faces = self._ensure_batched_faces(faces, batch_size, device)
        meshes = Meshes(verts=vertices, faces=faces)

        if cameras is None:
            cameras = self.create_camera(
                focal_length=focal_length,
                principal_point=principal_point,
                translation=translation,
                batch_size=batch_size,
                device=device,
            )

        return self.renderer(meshes, cameras=cameras)

    def render_alpha(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        focal_length: TensorLike = 1500.0,
        principal_point: Optional[TensorLike] = None,
        translation: Optional[Union[Sequence[float], torch.Tensor]] = None,
        cameras=None,
        channel_first: bool = True,
    ) -> torch.Tensor:
        """
        Render only the alpha/silhouette channel.

        Returns:
            [B,1,H,W] when channel_first=True, else [B,H,W].
        """
        rgba = self.render(vertices, faces, focal_length, principal_point, translation, cameras=cameras)
        alpha = rgba[..., 3]
        return alpha.unsqueeze(1) if channel_first else alpha

    def render_hard_mask(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        focal_length: TensorLike = 1500.0,
        principal_point: Optional[TensorLike] = None,
        translation: Optional[Union[Sequence[float], torch.Tensor]] = None,
        threshold: float = 0.5,
        cameras=None,
    ) -> torch.Tensor:
        """Convenience helper returning a non-differentiable hard mask [B,1,H,W]."""
        with torch.no_grad():
            alpha = self.render_alpha(vertices, faces, focal_length, principal_point, translation, cameras=cameras)
            return (alpha >= float(threshold)).to(alpha.dtype)

    def rasterize_fragments(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        focal_length: TensorLike = 1500.0,
        principal_point: Optional[TensorLike] = None,
        translation: Optional[Union[Sequence[float], torch.Tensor]] = None,
        cameras=None,
    ):
        """
        Return PyTorch3D rasterizer fragments for diagnostics/visibility checks.
        Useful for z-buffer/face visibility debugging.
        """
        vertices = self._ensure_batched_vertices(vertices)
        device = vertices.device
        batch_size = int(vertices.shape[0])
        faces = self._ensure_batched_faces(faces, batch_size, device)
        meshes = Meshes(verts=vertices, faces=faces)
        if cameras is None:
            cameras = self.create_camera(
                focal_length=focal_length,
                principal_point=principal_point,
                translation=translation,
                batch_size=batch_size,
                device=device,
            )
        return self.rasterizer(meshes, cameras=cameras)

    def to(self, device: Union[str, torch.device]) -> "SilhouetteRenderer":
        """Move renderer defaults to a new device and rebuild static renderer modules."""
        self.device = torch.device(device)
        self._build_renderer()
        return self

    def extra_repr(self) -> str:
        return (
            f"image_size={self.image_size}, sigma={self.sigma}, gamma={self.gamma}, "
            f"faces_per_pixel={self.faces_per_pixel}, bin_size={self.bin_size}, "
            f"max_faces_per_bin={self.max_faces_per_bin}"
        )


__all__ = ["SilhouetteRenderer"]
