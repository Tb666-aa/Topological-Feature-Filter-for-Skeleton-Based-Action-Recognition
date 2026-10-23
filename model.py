import math
import torch
from torch import nn
from .features import SkeletonFeatures, TopologicalFeatures
from .graph import multiscale_adjacency
from .kde import GaussianKDE
from .contact import ContactConsistency


class SpatialMultiScaleGCN(nn.Module):
    def __init__(self, cin, cout, scales=3):
        super().__init__()
        self.register_buffer("adjacency", multiscale_adjacency(scales))
        self.mask = nn.Parameter(torch.zeros_like(self.adjacency))
        self.project = nn.Conv2d(cin * scales, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = torch.einsum("nctv,kvw->nkctw", x, self.adjacency + self.mask)
        y = y.reshape(x.shape[0], -1, x.shape[2], x.shape[3])
        return self.bn(self.project(y))


class TemporalMultiBranchConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        branch_channels = math.ceil(cout / 7)

        def projection():
            return [nn.Conv2d(cin, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True)]

        branches = []
        for dilation in (1, 2, 3, 4):
            branches.append(nn.Sequential(*projection(), nn.Conv2d(
                branch_channels, branch_channels, (3, 1), stride=(stride, 1),
                padding=(dilation, 0), dilation=(dilation, 1), bias=False),
                nn.BatchNorm2d(branch_channels)))
        branches.append(nn.Sequential(*projection(), nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
                                      nn.BatchNorm2d(branch_channels)))
        branches.append(nn.Sequential(*projection(), nn.AvgPool2d((3, 1), (stride, 1), (1, 0)),
                                      nn.BatchNorm2d(branch_channels)))
        branches.append(nn.Sequential(nn.Conv2d(cin, branch_channels, 1, (stride, 1), bias=False),
                                      nn.BatchNorm2d(branch_channels)))
        self.branches = nn.ModuleList(branches)
        self.merge = nn.Sequential(nn.Conv2d(7 * branch_channels, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.merge(torch.cat([branch(x) for branch in self.branches], dim=1))


class SpatioTemporalBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.spatial = SpatialMultiScaleGCN(cin, cout)
        self.temporal = TemporalMultiBranchConv(cout, cout, stride)
        self.residual = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, (stride, 1), bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.temporal(self.relu(self.spatial(x))) + self.residual(x))


class TFFModel(nn.Module):
    def __init__(self, classes=60, channels=(64, 128, 256), bandwidth=0.1,
                 threshold=0.3, max_samples=256, topk=5, density_mode="density",
                 ph_frames=8, ph_points=96, time_scale=0.2,
                 verify_threshold=0.05, fusion_weights=(1.0, 0.5, 0.5), dropout=0.1,
                 contact_radius=0.35, contact_temperature=0.25, contact_threshold=0.3):
        super().__init__()
        if threshold <= 0 or topk < 1 or density_mode not in ("density", "posterior"):
            raise ValueError("Invalid filtering configuration")
        if len(channels) != 3 or len(fusion_weights) != 3 or min(fusion_weights) < 0 or sum(fusion_weights) <= 0:
            raise ValueError("Expected three channels and three nonnegative fusion weights")
        if not 0 <= verify_threshold <= 1:
            raise ValueError("verify_threshold must be in [0,1]")
        self.classes = classes
        self.threshold, self.topk, self.density_mode = threshold, topk, density_mode
        self.verify_threshold = verify_threshold
        self.contact_threshold = contact_threshold
        self.contacts = ContactConsistency(classes, contact_radius, contact_temperature)
        self.features = SkeletonFeatures()
        self.topology = TopologicalFeatures(ph_frames, ph_points, time_scale)
        self.kde = GaussianKDE(classes, bandwidth=bandwidth, max_samples=max_samples)
        self.topology_encoder = nn.Sequential(nn.Linear(10, 64), nn.LayerNorm(64), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(64, channels[0]), nn.Sigmoid())
        self.input_projection = nn.Sequential(nn.Conv2d(14, channels[0], 1, bias=False),
                                              nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(SpatioTemporalBlock(channels[0], channels[0]),
                                    SpatioTemporalBlock(channels[0], channels[1], 2),
                                    SpatioTemporalBlock(channels[1], channels[2], 2))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels[2], classes))
        self.auxiliary = nn.Linear(64, classes)
        self.register_buffer("fusion_weights", torch.tensor(fusion_weights, dtype=torch.float32))

    def encode_topology(self, descriptor):
        normalized = (descriptor - self.kde.mean) / self.kde.scale
        return self.topology_encoder(normalized)

    def backbone(self, x, embedding=None, reliable=None):
        present = x.abs().sum((1, 2, 3)) > 0
        x = self.features(x)
        n, c, t, v, m = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(n * m, c, t, v)
        x = self.input_projection(x)
        if embedding is not None:
            modulation = self.gate(embedding)
            if reliable is not None:
                modulation = modulation * reliable[:, None]
            modulation = modulation[:, None].expand(-1, m, -1).reshape(n * m, -1, 1, 1)
            x = x * (1 + modulation)
        x = self.blocks(x).mean((2, 3)).reshape(n, m, -1)
        x = (x * present.unsqueeze(-1)).sum(1) / present.sum(1, keepdim=True).clamp_min(1)
        return self.classifier(x)

    def forward(self, x, descriptor=None):
        descriptor = self.topology(x) if descriptor is None else descriptor
        embedding = self.encode_topology(descriptor)
        auxiliary = self.auxiliary(embedding)
        reliable = None
        if self.kde.fitted:
            with torch.no_grad():
                density = self.kde.log_density(descriptor)
                score = density if self.density_mode == "density" else density.log_softmax(-1)
                reliable = score.gather(1, auxiliary.argmax(-1, keepdim=True)).squeeze(1) >= math.log(self.threshold)
                contact_score = self.contacts.scores(self.contacts(x))
                reliable &= contact_score.gather(1, auxiliary.argmax(-1, keepdim=True)).squeeze(1) >= self.contact_threshold
        return self.backbone(x, embedding, reliable), auxiliary

    def fuse(self, backbone, auxiliary, topology):
        probabilities = torch.stack((backbone, auxiliary, topology), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        confidence = (1 - entropy / math.log(max(self.classes, 2))).clamp_min(1e-4)
        weights = confidence * self.fusion_weights[None]
        weights = weights / weights.sum(-1, keepdim=True)
        return (probabilities * weights.unsqueeze(-1)).sum(1), weights

    @torch.no_grad()
    def predict(self, x, repeated=True, prefix_frames=0, descriptor=None):
        self.eval()
        descriptor = self.topology(x) if descriptor is None else descriptor
        embedding = self.encode_topology(descriptor)
        auxiliary_probability = self.auxiliary(embedding).softmax(-1)
        log_density = self.kde.log_density(descriptor)
        topology_probability = log_density.softmax(-1)
        score = log_density if self.density_mode == "density" else log_density.log_softmax(-1)
        contact_score = self.contacts.scores(self.contacts(x))
        verified_topology = topology_probability * contact_score
        verified_topology = verified_topology / verified_topology.sum(-1, keepdim=True).clamp_min(1e-12)
        candidate = auxiliary_probability.argmax(-1)
        accepted = score.gather(1, candidate[:, None]).squeeze(1) >= math.log(self.threshold)
        accepted &= contact_score.gather(1, candidate[:, None]).squeeze(1) >= self.contact_threshold
        initial_accepted = accepted.clone()
        rounds = torch.ones_like(candidate)
        stage = accepted.long()  # 1: initial; 2: alternative; 3: fallback.
        if repeated:
            pending = ~accepted
            tried = torch.zeros_like(topology_probability, dtype=torch.bool)
            tried.scatter_(1, candidate[:, None], True)
            for step in range(min(self.topk, self.classes - 1)):
                available = ~tried
                ranked_score = (verified_topology * auxiliary_probability).masked_fill(~available, -1)
                alternative = ranked_score.argmax(-1)
                tried.scatter_(1, alternative[:, None], True)
                match = (pending
                         & (score.gather(1, alternative[:, None]).squeeze(1) >= math.log(self.threshold))
                         & (contact_score.gather(1, alternative[:, None]).squeeze(1) >= self.contact_threshold)
                         & (auxiliary_probability.gather(1, alternative[:, None]).squeeze(1) >= self.verify_threshold))
                rounds[pending] = step + 2
                candidate[match] = alternative[match]
                stage[match] = 2
                accepted |= match
                pending &= ~match
            stage[pending] = 3
        logits = self.backbone(x, embedding, accepted)
        backbone_probability = logits.softmax(-1)
        probability, weights = self.fuse(backbone_probability, auxiliary_probability, verified_topology)
        if not repeated:
            probability = backbone_probability
        unresolved = ~accepted
        if repeated and prefix_frames > 0 and unresolved.any():
            prefix = x[unresolved, :, :min(prefix_frames, x.shape[2])]
            prefix_descriptor = self.topology(prefix)
            prefix_embedding = self.encode_topology(prefix_descriptor)
            prefix_probability = self.backbone(prefix, prefix_embedding).softmax(-1)
            probability[unresolved] = 0.5 * (probability[unresolved] + prefix_probability)
        return {"prediction": probability.argmax(-1), "probability": probability,
                "logits": logits, "accepted": accepted, "initial_accepted": initial_accepted,
                "candidate": candidate, "stage": stage, "rounds": rounds,
                "fusion_weights": weights, "topology_probability": topology_probability,
                "contact_consistency": contact_score}
