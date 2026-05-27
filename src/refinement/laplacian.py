import torch


class LaplacianRegularizer:
    """
    Fast edge-based smoothness regularizer.

    For freeform refinement, regularizing offsets is usually safer than
    regularizing absolute vertices because it preserves the SMPL-X surface.
    """

    def __init__(self, faces, device="cuda"):
        self.device = device

        if isinstance(faces, torch.Tensor):
            f = faces.detach().long().cpu()
        else:
            f = torch.tensor(faces, dtype=torch.long)

        if f.dim() == 3:
            f = f[0]

        edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=1).values
        edges = torch.unique(edges, dim=0)
        self.edges = edges.to(device=device, dtype=torch.long)

    def edge_smoothness(self, values):
        if values.dim() == 2:
            values = values.unsqueeze(0)
        a = values[:, self.edges[:, 0]]
        b = values[:, self.edges[:, 1]]
        return ((a - b) ** 2).mean()

    def offset_loss(self, offsets):
        return self.edge_smoothness(offsets)

    def surface_loss(self, vertices):
        return self.edge_smoothness(vertices)
