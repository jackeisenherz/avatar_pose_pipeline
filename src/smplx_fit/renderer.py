import torch

from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
)


class SilhouetteRenderer:

    def __init__(
        self,
        image_size=1024,
        device="cuda"
    ):
        self.device = device
        self.image_size = image_size

        self.blend_params = BlendParams(
            sigma=1e-4,
            gamma=1e-4
        )

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=50
        )

    def render(
        self,
        vertices,
        faces,
        focal_length,
        principal_point,
        translation
    ):

        batch_size = vertices.shape[0]

        R = torch.eye(
            3,
            device=self.device
        ).unsqueeze(0).repeat(batch_size, 1, 1)

        T = translation

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            image_size=torch.tensor(
                [[self.image_size, self.image_size]],
                device=self.device
            ),
            in_ndc=False,
            device=self.device
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )

        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftSilhouetteShader(
                blend_params=self.blend_params
            )
        )

        meshes = Meshes(
            verts=vertices,
            faces=faces
        )

        silhouettes = renderer(
            meshes,
            cameras=cameras
        )

        return silhouettes

    def create_camera(self):

        from pytorch3d.renderer import (
            PerspectiveCameras
        )

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

        return PerspectiveCameras(
            focal_length=focal,
            principal_point=center,
            image_size=[[
                self.image_size,
                self.image_size
            ]],
            in_ndc=False,
            device=self.device
        )