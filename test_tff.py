"""Numerical, routing and end-to-end smoke tests; no dataset download required."""
import io
import math
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from tff import TFFModel, GaussianKDE
from tff.data import preprocess
from tff.features import TopologicalFeatures
from tff.persistence import PersistentHomology
from tff.calibration import select_threshold


class TFFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(2, 3, 8, 25, 2)
        self.x[..., 1] = 0
        self.model = TFFModel(classes=3, channels=(7, 14, 28), ph_frames=2, ph_points=32)

    def test_persistence_detects_circle(self):
        ph = PersistentHomology(max_points=64)
        angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        circle = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(48)))
        line = np.column_stack((np.linspace(0, 1, 48), np.zeros((48, 2))))
        self.assertGreater(ph.cloud_summary(circle)[3], 1.0)
        self.assertAlmostEqual(float(ph.cloud_summary(line)[3]), 0.0)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(ph.cloud_summary(circle), ph.cloud_summary(circle @ rotation + 3), atol=1e-5)

    def test_topology_and_empty_input(self):
        topology = self.model.topology(self.x)
        self.assertEqual(tuple(topology.shape), (2, 10))
        self.assertTrue(torch.isfinite(topology).all())
        self.assertTrue(torch.equal(self.model.topology(torch.zeros_like(self.x)), torch.zeros(2, 10)))

    def test_missing_person_does_not_change_topology(self):
        one = self.model.topology(self.x[..., :1])
        padded = self.model.topology(self.x)
        torch.testing.assert_close(one, padded, atol=1e-5, rtol=1e-5)

    def test_shared_origin_preserves_interactions(self):
        raw = np.ones((3, 4, 25, 2), dtype=np.float32)
        raw[..., 1] += 2
        out = preprocess(raw, 4)
        torch.testing.assert_close(out[:, :, 0, 1] - out[:, :, 0, 0], torch.full((3, 4), 2.0))

    def test_kde_matches_gaussian(self):
        kde = GaussianKDE(2, dimensions=1, bandwidth=0.1, max_samples=2)
        kde.fit(torch.tensor([[0.0], [2.0]]), torch.tensor([0, 1]))
        density = kde.log_density(torch.tensor([[0.0]]))
        self.assertAlmostEqual(density[0, 0].item(), -math.log(0.1 * math.sqrt(2 * math.pi)), places=5)
        self.assertEqual(kde(torch.tensor([[0.0]])).argmax(-1).item(), 0)

    def test_backward_and_checkpoint(self):
        descriptor = self.model.topology(self.x)
        self.model.kde.fit(descriptor, torch.tensor([0, 1]))
        logits, auxiliary = self.model(self.x, descriptor)
        loss = F.cross_entropy(logits, torch.tensor([0, 1])) + F.cross_entropy(auxiliary, torch.tensor([0, 1]))
        loss.backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertIsNotNone(self.model.gate[0].weight.grad)
        self.model.eval()
        before = self.model.predict(self.x, descriptor=descriptor, prefix_frames=4)['probability']
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        restored = TFFModel(classes=3, channels=(7, 14, 28), ph_frames=2, ph_points=32)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        after = restored.predict(self.x, descriptor=descriptor, prefix_frames=4)['probability']
        torch.testing.assert_close(before, after)

    def test_rejected_initial_candidate_can_accept_alternative(self):
        descriptor = self.model.topology(self.x)
        self.model.kde.fit(descriptor, torch.tensor([0, 1]))
        with torch.no_grad():
            self.model.auxiliary.weight.zero_()
            self.model.auxiliary.bias.copy_(torch.tensor([2.0, 1.0, 0.0]))
        density = torch.tensor([[-5.0, 0.0, -10.0], [-5.0, 0.0, -10.0]])
        with patch.object(self.model.kde, 'log_density', return_value=density):
            result = self.model.predict(self.x, descriptor=descriptor)
        self.assertFalse(result['initial_accepted'].any())
        self.assertTrue(result['accepted'].all())
        self.assertTrue((result['candidate'] == 1).all())
        self.assertTrue((result['stage'] == 2).all())

    def test_fallback_prefix_and_probabilities(self):
        descriptor = self.model.topology(self.x)
        self.model.kde.fit(descriptor, torch.tensor([0, 1]))
        self.model.threshold = 1e30
        result = self.model.predict(self.x, descriptor=descriptor, prefix_frames=4)
        self.assertFalse(result['accepted'].any())
        self.assertTrue((result['stage'] == 3).all())
        torch.testing.assert_close(result['probability'].sum(-1), torch.ones(2))

    def test_threshold_calibration(self):
        best, _ = select_threshold(torch.tensor([0.9, 0.8, 0.1]),
                                   torch.tensor([True, True, False]), [0.05, 0.5, 0.85])
        self.assertEqual(best['threshold'], 0.5)
        self.assertEqual(best['f1'], 1.0)

    def test_contact_consistency(self):
        values = self.model.contacts(self.x)
        self.assertEqual(tuple(values.shape), (2, 6))
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        references = torch.stack((torch.zeros(6), torch.ones(6)))
        self.model.contacts.fit(references, torch.tensor([0, 1]))
        scores = self.model.contacts.scores(references)
        self.assertEqual(scores.argmax(-1).tolist(), [0, 1])
        self.assertTrue((scores[:, 2] == 0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
