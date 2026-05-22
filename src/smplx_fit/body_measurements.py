import torch


class BodyMeasurements:

    """
    Approximate vertex regions
    for body anatomical constraints.
    """

    CHEST_VERTICES = [
        3200, 3300, 3400, 3500
    ]

    WAIST_VERTICES = [
        4200, 4300, 4400
    ]

    HIP_VERTICES = [
        5200, 5300, 5400
    ]

    @staticmethod
    def width(vertices, indices):

        verts = vertices[:, indices]

        x = verts[:, :, 0]

        width = (
            x.max(dim=1).values -
            x.min(dim=1).values
        )

        return width