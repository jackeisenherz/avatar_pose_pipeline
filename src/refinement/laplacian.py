import torch


class LaplacianRegularizer:

    def __init__(self, faces, num_vertices, device="cuda"):

        self.device = device

        self.neighbors = self._build_neighbors(
            faces,
            num_vertices
        )

    def _build_neighbors(
        self,
        faces,
        num_vertices
    ):

        neighbors = [
            set()
            for _ in range(num_vertices)
        ]

        faces = faces.cpu().numpy()

        for f in faces:

            a, b, c = f

            neighbors[a].update([b, c])
            neighbors[b].update([a, c])
            neighbors[c].update([a, b])

        return [
            torch.tensor(
                list(n),
                dtype=torch.long,
                device=self.device
            )
            for n in neighbors
        ]

    def loss(self, vertices):

        total = 0.0

        for vidx, nbrs in enumerate(self.neighbors):

            if len(nbrs) == 0:
                continue

            v = vertices[:, vidx]

            nbr_mean = vertices[:, nbrs].mean(dim=1)

            total += (
                (v - nbr_mean) ** 2
            ).mean()

        return total / len(self.neighbors)