import sys
sys.path.insert(0, "/home/jaewan/Genie")

import torch
import torch.nn as nn

from genie.semantic.hooks import HookManager


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_hookmanager_captures_module_path_and_shapes():
    model = Tiny().eval()
    hm = HookManager()
    hm.inject_hooks(model)

    x = torch.randn(3, 8)
    _ = model(x)

    ctx = hm.get_context()
    # Should have entries for at least fc1, relu, fc2
    assert any(k.endswith("fc1") for k in ctx.keys())
    assert any(k.endswith("relu") for k in ctx.keys())
    assert any(k.endswith("fc2") for k in ctx.keys())

    # Check shape capture format
    for info in ctx.values():
        assert "module_path" in info
        assert isinstance(info.get("input_shapes"), list)
        assert "output_shape" in info


