"""Test that semantic metadata is captured during graph construction."""

import torch
import torch.nn as nn
import djinn


def test_metadata_captured_in_linear():
    """Test metadata capture in simple linear layer."""

    # Genie initializes automatically on import

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    # Use capture context to create LazyTensors
    with genie.capture():
        # Create LazyTensor input
        x = torch.randn(5, 10)

        # Forward pass (should capture metadata)
        y = model(x)

    # Check that metadata was captured
    assert hasattr(y, 'metadata'), "LazyTensor should have metadata"
    assert y.metadata != {}, "Metadata should not be empty"

    # Check expected fields
    if 'module_path' in y.metadata:
        print(f"✓ Module path captured: {y.metadata['module_path']}")

    if 'semantic_role' in y.metadata:
        print(f"✓ Semantic role: {y.metadata['semantic_role']}")

    print(f"Full metadata: {y.metadata}")


def test_metadata_captured_in_attention():
    """Test metadata capture in attention-like pattern."""

    # Genie initializes automatically on import

    # Simulate attention: Q @ K.T → softmax → @ V
    # Create LazyTensors using the remote device
    Q = torch.randn(32, 64, 128, device='remote_accelerator:0')
    K = torch.randn(32, 64, 128, device='remote_accelerator:0')
    V = torch.randn(32, 64, 128, device='remote_accelerator:0')

    # Q @ K.T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Check metadata
    assert 'semantic_role' in scores.metadata
    print(f"✓ Scores metadata: {scores.metadata}")

    # Softmax
    attn = torch.softmax(scores, dim=-1)
    assert 'semantic_role' in attn.metadata
    print(f"✓ Attention metadata: {attn.metadata}")

    # @ V
    output = torch.matmul(attn, V)

    # Check for pattern hints
    if 'pattern_hints' in output.metadata:
        hints = output.metadata['pattern_hints']
        assert hints.get('likely_pattern') == 'attention', \
            "Should detect attention pattern"
        print(f"✓ Pattern hint detected: {hints}")


if __name__ == '__main__':
    test_metadata_captured_in_linear()
    test_metadata_captured_in_attention()
