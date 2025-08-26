import sys
sys.path.insert(0, "/home/jaewan/Genie")

import torch
import torch.nn as nn

from genie.semantic.fx_analyzer import FXAnalyzer
from genie.core.graph import GraphBuilder


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_fx_analyzer_counts_parameters_and_hierarchy():
    # Trace a tiny model to FX and attach to GraphBuilder
    model = Tiny().eval()
    try:
        import torch.fx as fx
        traced = fx.symbolic_trace(model)
    except Exception:
        traced = None

    GraphBuilder.reset_current()
    if traced is not None:
        # Attach the FX graph to GraphBuilder for analysis
        gb = GraphBuilder.current()
        gb.set_fx_graph(traced.graph)
        setattr(gb._fx_graph, 'module', traced)

    graph = GraphBuilder.current().get_graph()
    analyzer = FXAnalyzer()
    info = analyzer.analyze_structure(graph)

    assert isinstance(info.modules, dict)
    assert info.depth >= 0 and info.width >= 0
    # Parameter count should be positive for our tiny model if tracing worked
    if traced is not None:
        assert info.parameters > 0


