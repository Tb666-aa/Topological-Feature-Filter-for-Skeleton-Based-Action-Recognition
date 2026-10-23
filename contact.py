"""Contact-state descriptors and class-conditional consistency verification."""
import torch
from torch import nn
from .features import joint_validity


class ContactConsistency(nn.Module):
    pairs = ((7, 11), (7, 3), (11, 3), (7, 0), (11, 0))

    def __init__(self, classes, radius=0.35, temperature=0.25):
        super().__init__()
        if radius <= 0 or temperature <= 0:
            raise ValueError("Contact radius and temperature must be positive")
        self.radius, self.temperature = radius, temperature
        self.register_buffer("prototypes", torch.zeros(classes, 6))
        self.register_buffer("counts", torch.zeros(classes, dtype=torch.long))

    @torch.no_grad()
    def forward(self, x):
        valid = joint_validity(x)
        torso = (x[..., 20, :] - x[..., 0, :]).norm(dim=1).clamp_min(1e-4)
        values = []
        for a, b in self.pairs:
            mask = valid[:, :, a] & valid[:, :, b]
            distance = (x[..., a, :] - x[..., b, :]).norm(dim=1) / torso
            values.append(((distance < self.radius) * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1))
        if x.shape[-1] > 1:
            # Minimum cross-person wrist distance in each frame.
            left = x[:, :, :, [7, 11], 0].permute(0, 2, 3, 1)
            right = x[:, :, :, [7, 11], 1].permute(0, 2, 3, 1)
            distance = torch.cdist(left, right)
            pair_mask = valid[:, :, [7, 11], 0].unsqueeze(-1) & valid[:, :, [7, 11], 1].unsqueeze(-2)
            distance = distance.masked_fill(~pair_mask, torch.inf).flatten(2).amin(-1)
            mask = pair_mask.flatten(2).any(-1)
            scale = torso[:, :, :2].mean(-1)
            values.append(((distance / scale < self.radius) * mask).sum(-1) / mask.sum(-1).clamp_min(1))
        else:
            values.append(x.new_zeros(len(x)))
        return torch.stack(values, dim=-1)

    @torch.no_grad()
    def fit(self, descriptors, labels):
        self.counts.zero_()
        self.prototypes.zero_()
        self.prototypes.index_add_(0, labels, descriptors)
        self.counts.add_(torch.bincount(labels, minlength=len(self.counts)))
        self.prototypes.div_(self.counts[:, None].clamp_min(1))

    def scores(self, descriptor):
        if not bool(self.counts.any()):
            return descriptor.new_ones((len(descriptor), len(self.counts)))
        error = (descriptor[:, None] - self.prototypes[None]).abs().mean(-1)
        score = torch.exp(-error / self.temperature)
        return score.masked_fill(self.counts[None] == 0, 0)
