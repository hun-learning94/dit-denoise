"""CPU checks of the exported intervention and metric definitions."""
import unittest
import json
from pathlib import Path
import numpy as np
import torch

from experiments.component_transfer_scale.model import hybrid_state, LoRALinear
from experiments.uvit_metfaces.evaluate import metrics
from experiments.norm_matched.worker import matched, norm


class RevisionTests(unittest.TestCase):
    def test_transfer_horizons_and_native_views(self):
        root = Path(__file__).resolve().parents[1]
        settings = []
        for path in (root / 'revision/configs').glob('transfer_*.json'):
            manifest = json.loads(path.read_text())
            for row in manifest['runs']:
                expected = 100 if 'ffhq' in row['target'].lower() else 200
                self.assertEqual(row['reference_epochs'], expected)
                self.assertNotIn('imagenet', row['target'].lower())
                self.assertNotIn('continuation', row)
                settings.append((row['family'], row['target']))
        self.assertEqual(len(settings), 9)

    def test_paired_state_boundaries_and_fused_qkv(self):
        source = {'blocks.0.attn.qkv.weight': torch.zeros(6, 2),
                  'blocks.0.attn.proj.weight': torch.zeros(2, 2),
                  'blocks.0.mlp.fc1.weight': torch.zeros(8, 2),
                  'blocks.0.mlp.fc2.weight': torch.zeros(2, 8),
                  'blocks.0.attn.qkv.bias': torch.zeros(6)}
        adapted = {k: (torch.ones_like(v) * (3 if '.mlp.' in k else 2))
                   if k.endswith('.weight') else v.clone() for k, v in source.items()}
        adapted['blocks.0.attn.qkv.weight'][:4] = 1
        for bits, endpoint in [('000', source), ('111', adapted)]:
            result = hybrid_state(source, adapted, bits)
            for name in result:
                torch.testing.assert_close(result[name], endpoint[name], rtol=0, atol=0)
        qk = hybrid_state(source, adapted, '100')
        torch.testing.assert_close(qk['blocks.0.attn.qkv.weight'][:4], adapted['blocks.0.attn.qkv.weight'][:4])
        torch.testing.assert_close(qk['blocks.0.attn.qkv.weight'][4:], source['blocks.0.attn.qkv.weight'][4:])
        state, info = matched(source, adapted, '011', hybrid_state)
        self.assertAlmostEqual(norm(state, source), norm(qk, source), places=5)
        self.assertLess(info['scale'], 1)
        torch.testing.assert_close(state['blocks.0.attn.qkv.weight'][:4], source['blocks.0.attn.qkv.weight'][:4])

    def test_zero_lora_update_and_effective_weights(self):
        torch.manual_seed(7)
        original = torch.nn.Linear(4, 7).requires_grad_(False)
        adapter = LoRALinear(original, rank=2)
        x = torch.randn(3, 4)
        torch.testing.assert_close(adapter(x), original(x), rtol=0, atol=0)
        with torch.no_grad():
            adapter.lora_B.normal_()
        wanted = torch.nn.functional.linear(x, adapter.effective(), adapter.bias)
        torch.testing.assert_close(adapter(x), wanted)

    def test_exact_kid_and_small_reference_fid(self):
        from scipy.linalg import sqrtm
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(17, 8)), rng.normal(size=(21, 8))
        result = metrics(x, y)
        xx, yy, xy = (x @ x.T / 8 + 1)**3, (y @ y.T / 8 + 1)**3, (x @ y.T / 8 + 1)**3
        expected = (xx.sum() - np.trace(xx)) / (17 * 16) + (yy.sum() - np.trace(yy)) / (21 * 20) - 2 * xy.mean()
        self.assertAlmostEqual(result['kid'], expected, places=10)
        cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        expected_fid = ((x.mean(0) - y.mean(0))**2).sum() + np.trace(cx + cy - 2 * sqrtm(cx @ cy)).real
        self.assertAlmostEqual(result['fid'], expected_fid, places=9)


if __name__ == '__main__':
    unittest.main()
