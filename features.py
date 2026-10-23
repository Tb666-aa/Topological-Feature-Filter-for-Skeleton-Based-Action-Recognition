import torch
from torch import nn
from .graph import EDGES, PARENTS
from .persistence import PersistentHomology


def joint_validity(x):
    valid = x.abs().sum(1) > 0
    # Translation maps a valid root to zero; recover its body-presence mask.
    valid[:, :, 0] = valid.any(dim=2)
    return valid


def temporal_difference(x, valid=None):
    difference = x[:, :, 1:] - x[:, :, :-1]
    if valid is not None:
        difference = difference * (valid[:, 1:] & valid[:, :-1]).unsqueeze(1)
    return torch.cat((difference, torch.zeros_like(x[:, :, :1])), dim=2)


def smooth_angles(theta):
    if theta.shape[1] < 3:
        return theta
    padded = torch.cat((theta[:, :1], theta, theta[:, -1:]), dim=1)
    median = padded.unfold(1, 3, 1).median(dim=-1).values
    delta = torch.cat((torch.zeros_like(theta[:, :1]), theta[:, 1:] - theta[:, :-1]), dim=1)
    return torch.where(delta.abs() > torch.pi / 4, median, theta)


class SkeletonFeatures(nn.Module):
    """Joint, bone, AGE, joint velocity, bone velocity and angular velocity."""
    def __init__(self):
        super().__init__()
        self.register_buffer("parents", torch.tensor(PARENTS))
        children = {i: [] for i in range(25)}
        for child, parent in EDGES:
            children[parent].append(child)
        endpoint1, endpoint2 = [], []
        for joint in range(25):
            if PARENTS[joint] == joint:
                a, b = children[joint][:2]
            else:
                a = PARENTS[joint]
                b = children[joint][0] if children[joint] else PARENTS[a]
            endpoint1.append(a)
            endpoint2.append(b)
        self.register_buffer("endpoint1", torch.tensor(endpoint1))
        self.register_buffer("endpoint2", torch.tensor(endpoint2))

    def forward(self, x):
        valid = joint_validity(x)
        bone_valid = valid & valid.index_select(2, self.parents)
        bone = (x.index_select(3, self.parents) - x) * bone_valid.unsqueeze(1)
        a = x.index_select(3, self.endpoint1) - x
        b = x.index_select(3, self.endpoint2) - x
        norm_a, norm_b = a.norm(dim=1), b.norm(dim=1)
        angle_valid = (valid & valid.index_select(2, self.endpoint1)
                       & valid.index_select(2, self.endpoint2) & (norm_a > 1e-8) & (norm_b > 1e-8))
        cosine = ((a * b).sum(1) / (norm_a * norm_b).clamp_min(1e-8)).clamp(-1, 1)
        theta = smooth_angles(cosine.acos())
        age = torch.where(angle_valid, 1 - theta.cos(), 0).unsqueeze(1)
        return torch.cat((x, bone, age, temporal_difference(x, valid),
                          temporal_difference(bone, bone_valid),
                          temporal_difference(age, angle_valid)), dim=1)


class TopologicalFeatures(nn.Module):
    """Five geometric statistics concatenated with five persistence statistics."""
    names = ("angle_mean", "angle_std", "nonadjacent_distance_mean",
             "nonadjacent_distance_std", "arm_bone_ratio", "h0_mean_lifetime",
             "h0_max_lifetime", "h1_total_persistence", "h1_max_lifetime", "h1_entropy")
    angle_triplets = ((5, 4, 6), (9, 8, 10), (13, 12, 14), (17, 16, 18))

    def __init__(self, sampled_frames=8, max_points=96, time_scale=0.2):
        super().__init__()
        self.persistence = PersistentHomology(sampled_frames, max_points, time_scale)
        adjacent = {tuple(sorted(edge)) for edge in EDGES}
        pairs = [(a, b) for a in range(25) for b in range(a + 1, 25)
                 if (a, b) not in adjacent]
        self.register_buffer("pair_a", torch.tensor([a for a, _ in pairs]))
        self.register_buffer("pair_b", torch.tensor([b for _, b in pairs]))

    @staticmethod
    def moments(values, mask):
        values, mask = values.flatten(1), mask.flatten(1).to(values.dtype)
        count = mask.sum(-1).clamp_min(1)
        mean = (values * mask).sum(-1) / count
        variance = ((values - mean[:, None]).square() * mask).sum(-1) / count
        return mean, variance.sqrt()

    @torch.no_grad()
    def forward(self, x):
        valid = joint_validity(x)
        angles, masks = [], []
        for u, a, b in self.angle_triplets:
            va, vb = x[..., a, :] - x[..., u, :], x[..., b, :] - x[..., u, :]
            na, nb = va.norm(dim=1), vb.norm(dim=1)
            cosine = ((va * vb).sum(1) / (na * nb).clamp_min(1e-8)).clamp(-1, 1)
            angles.append(cosine.acos())
            masks.append(valid[:, :, u] & valid[:, :, a] & valid[:, :, b] & (na > 1e-8) & (nb > 1e-8))
        angle_mean, angle_std = self.moments(torch.stack(angles, 2), torch.stack(masks, 2))
        distance = (x.index_select(3, self.pair_a) - x.index_select(3, self.pair_b)).norm(dim=1)
        pair_valid = valid.index_select(2, self.pair_a) & valid.index_select(2, self.pair_b)
        distance_mean, distance_std = self.moments(distance, pair_valid)
        ratios = []
        for a, b, c in ((4, 5, 6), (8, 9, 10)):
            mask = valid[:, :, a] & valid[:, :, b] & valid[:, :, c]
            upper = (x[..., a, :] - x[..., b, :]).norm(dim=1)
            lower = (x[..., b, :] - x[..., c, :]).norm(dim=1)
            ratios.append((upper * mask).sum((1, 2)) / (lower * mask).sum((1, 2)).clamp_min(1e-8))
        geometry = torch.stack((angle_mean, angle_std, distance_mean, distance_std,
                                torch.stack(ratios).mean(0)), dim=-1)
        return torch.cat((geometry, self.persistence(x, valid)), dim=-1)
