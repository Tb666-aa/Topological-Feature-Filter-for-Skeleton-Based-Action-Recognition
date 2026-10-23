"""Vietoris-Rips persistence over Z/2Z for skeleton point clouds."""
import numpy as np
import torch
from torch import nn


class PersistentHomology(nn.Module):
    def __init__(self, sampled_frames=8, max_points=96, time_scale=0.2):
        super().__init__()
        if sampled_frames < 1 or max_points < 3 or time_scale < 0:
            raise ValueError("Invalid persistence configuration")
        self.sampled_frames = sampled_frames
        self.max_points = max_points
        self.time_scale = time_scale

    @staticmethod
    def summarize(diagrams):
        lifetimes = []
        for diagram in diagrams:
            finite = diagram[np.isfinite(diagram).all(axis=1)]
            lifetime = finite[:, 1] - finite[:, 0]
            lifetimes.append(lifetime[lifetime > 1e-8])
        h0, h1 = lifetimes
        total = h1.sum() if len(h1) else 0.0
        weights = h1 / total if total > 0 else np.zeros(0)
        entropy = -(weights * np.log(weights)).sum() if len(weights) else 0.0
        return np.array([h0.mean() if len(h0) else 0.0,
                         h0.max() if len(h0) else 0.0,
                         total, h1.max() if len(h1) else 0.0, entropy], dtype=np.float32)

    def cloud_summary(self, points):
        from ripser import ripser
        if len(points) < 3:
            return np.zeros(5, dtype=np.float32)
        points = np.unique(points, axis=0)
        if len(points) > self.max_points:
            selected = [0]
            distance = np.full(len(points), np.inf)
            for _ in range(1, self.max_points):
                distance = np.minimum(distance, ((points - points[selected[-1]]) ** 2).sum(-1))
                selected.append(int(distance.argmax()))
            points = points[selected]
        return self.summarize(ripser(points, maxdim=1, coeff=2)["dgms"])

    @torch.no_grad()
    def forward(self, x, valid):
        coordinates = x.detach().cpu().permute(0, 2, 3, 4, 1).numpy()
        masks = valid.detach().cpu().numpy()
        output = []
        for sequence, mask in zip(coordinates, masks):
            frames = np.flatnonzero(mask.any(axis=(1, 2)))
            if not len(frames):
                output.append(np.zeros(5, dtype=np.float32))
                continue
            indices = np.unique(np.linspace(0, len(frames) - 1,
                                           min(self.sampled_frames, len(frames))).round().astype(int))
            selected = frames[indices]
            static, spacetime = [], []
            for frame in selected:
                points = sequence[frame][mask[frame]]
                static.append(self.cloud_summary(points))
                time = self.time_scale * frame / max(len(sequence) - 1, 1)
                spacetime.append(np.column_stack((points, np.full(len(points), time))))
            dynamic = self.cloud_summary(np.concatenate(spacetime, axis=0))
            output.append(0.5 * (np.mean(static, axis=0) + dynamic))
        return torch.as_tensor(np.stack(output), device=x.device, dtype=x.dtype)
