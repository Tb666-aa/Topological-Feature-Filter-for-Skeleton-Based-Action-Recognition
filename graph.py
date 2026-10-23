import torch

# NTU RGB+D, zero-based joint indices; (child, parent).
EDGES = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
         (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
         (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16),
         (18, 17), (19, 18), (21, 22), (22, 7), (23, 24), (24, 11)]
PARENTS = list(range(25))
for child, parent in EDGES:
    PARENTS[child] = parent


def multiscale_adjacency(scales=3):
    """Independently normalized exact-hop neighborhoods."""
    adjacency = torch.zeros(25, 25)
    for a, b in EDGES:
        adjacency[a, b] = adjacency[b, a] = 1
    reached = torch.eye(25, dtype=torch.bool)
    power = torch.eye(25)
    result = []
    for _ in range(scales):
        neighborhood = (power > 0) & ~reached if result else reached.clone()
        reached |= power > 0
        matrix = neighborhood.float()
        degree = matrix.sum(-1).clamp_min(1).rsqrt()
        result.append(degree[:, None] * matrix * degree[None, :])
        power = power @ (adjacency + torch.eye(25))
    return torch.stack(result)
