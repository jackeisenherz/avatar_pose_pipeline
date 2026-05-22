# ============================================================
# FILE:
# src/photometric/rgb_renderer.py
# ============================================================

import torch

from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesUV,
    BlendParams
)


class RGBRenderer:

    def __init__(
        self,
        image_size=1024,
        device="cuda"
    ):

        self.device = device

        self.image_size = image_size

        self.lights = PointLights(
            location=[[0.0, 0.0, -3.0]],
            device=device
        )

        blend = BlendParams(
            sigma=1e-4,
            gamma=1e-4
        )

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                lights=self.lights,
                blend_params=blend
            )
        )

    def create_camera(self):

        focal = torch.tensor(
            [[1500.0, 1500.0]],
            device=self.device
        )

        center = torch.tensor(
            [[
                self.image_size / 2,
                self.image_size / 2
            ]],
            device=self.device
        )

        cameras = PerspectiveCameras(
            focal_length=focal,
            principal_point=center,
            image_size=[[
                self.image_size,
                self.image_size
            ]],
            in_ndc=False,
            device=self.device
        )

        return cameras

    def render(
        self,
        vertices,
        faces,
        verts_uvs,
        faces_uvs,
        texture_map,
        cameras
    ):

        textures = TexturesUV(
            maps=texture_map,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs
        )

        mesh = Meshes(
            verts=vertices,
            faces=faces.unsqueeze(0),
            textures=textures
        )

        rendered = self.renderer(
            mesh,
            cameras=cameras
        )

        return rendered[..., :3]