import math
import torch
from torch import nn


class GaussianKDE(nn.Module):
    """Class-conditional Gaussian KDE evaluated with log-sum-exp."""
    def __init__(self, classes, dimensions=10, bandwidth=0.1, max_samples=256):
        super().__init__()
        if bandwidth <= 0 or max_samples < 1:
            raise ValueError("bandwidth and max_samples must be positive")
        self.bandwidth = bandwidth
        self.classes = classes
        self.max_samples = max_samples
        self.register_buffer("references", torch.zeros(classes, max_samples, dimensions))
        self.register_buffer("counts", torch.zeros(classes, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dimensions))
        self.register_buffer("scale", torch.ones(dimensions))

    @property
    def fitted(self):
        return bool((self.counts > 0).any())

    @torch.no_grad()
    def fit(self, features, labels, seed=1):
        if features.ndim != 2 or features.shape[1] != self.mean.numel():
            raise ValueError("features must have shape (N, dimensions)")
        if len(features) == 0 or len(labels) != len(features):
            raise ValueError("empty features or mismatched labels")
        if not torch.isfinite(features).all() or labels.min() < 0 or labels.max() >= self.classes:
            raise ValueError("invalid features or labels")
        self.mean.copy_(features.mean(0))
        self.scale.copy_(features.std(0, unbiased=False).clamp_min(1e-4))
        features = (features - self.mean) / self.scale
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.counts.zero_()
        self.references.zero_()
        for c in range(self.classes):
            values = features[labels == c]
            indices = torch.randperm(len(values), generator=generator)[:self.max_samples].to(values.device)
            selected = values[indices]
            self.counts[c] = len(selected)
            self.references[c, :len(selected)].copy_(selected)

    def log_density(self, features):
        if not self.fitted:
            raise RuntimeError("Fit KDE using training data before inference")
        normalized = (features - self.mean) / self.scale
        normalizer = self.mean.numel() * math.log(self.bandwidth * math.sqrt(2 * math.pi))
        result = []
        for c in range(self.classes):
            n = int(self.counts[c])
            if not n:
                result.append(features.new_full((len(features),), -torch.inf))
                continue
            distance = torch.cdist(normalized, self.references[c, :n]).square()
            log_kernel = -distance / (2 * self.bandwidth ** 2)
            result.append(torch.logsumexp(log_kernel, dim=-1) - math.log(n) - normalizer)
        return torch.stack(result, dim=-1)

    def forward(self, features):
        return self.log_density(features).softmax(-1)
