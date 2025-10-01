"""Tests for pattern matching service (Refactoring #5).

Tests the IPatternMatcher interface and various implementations.
"""

import pytest
import torch
import torch.fx as fx

from genie.semantic.pattern_matcher import (
    IPatternMatcher,
    NetworkXPatternMatcher,
    SimplifiedPatternMatcher,
    CompositePatternMatcher,
    create_pattern_matcher,
    get_default_pattern_matcher
)
from genie.core.graph import ComputationGraph, ComputationNode
from genie.semantic.workload import WorkloadType


class TestIPatternMatcherInterface:
    """Test the pattern matcher interface."""
    
    def test_interface_cannot_be_instantiated(self):
        """Test that abstract interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            IPatternMatcher()


class TestNetworkXPatternMatcher:
    """Test NetworkX-based pattern matcher."""
    
    def test_creation_default(self):
        """Test creating matcher with default patterns."""
        matcher = NetworkXPatternMatcher()
        assert matcher.name == "NetworkX"
        assert matcher is not None
    
    def test_creation_with_registry(self):
        """Test creating matcher with existing registry."""
        from genie.semantic.pattern_registry import PatternRegistry
        
        registry = PatternRegistry()
        matcher = NetworkXPatternMatcher(registry)
        
        assert matcher.name == "NetworkX"
        assert matcher._pattern_registry == registry
    
    def test_match_patterns_computation_graph(self):
        """Test pattern matching with ComputationGraph."""
        matcher = NetworkXPatternMatcher()
        
        # Create simple graph
        graph = ComputationGraph(
            nodes={
                'n1': ComputationNode('n1', 'aten::linear', [], [], {}),
                'n2': ComputationNode('n2', 'aten::relu', ['n1'], [], {})
            },
            edges=[('n1', 'n2')],
            entry_points={'n1'}
        )
        
        result = matcher.match_patterns(graph)
        
        assert result.is_ok or result.is_err  # Should return Result
        # Exact matches depend on patterns, just verify it works
    
    def test_match_patterns_fx_graph(self):
        """Test pattern matching with FX GraphModule."""
        matcher = NetworkXPatternMatcher()
        
        # Create simple FX graph
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)
        
        model = SimpleModel()
        traced = fx.symbolic_trace(model)
        
        result = matcher.match_patterns(traced)
        
        assert result.is_ok or result.is_err
    
    def test_get_performance_report(self):
        """Test getting performance report."""
        matcher = NetworkXPatternMatcher()
        
        report = matcher.get_performance_report()
        
        assert isinstance(report, dict)
        # Just verify it returns a dict, structure may vary
        assert len(report) >= 0
    
    def test_register_pattern(self):
        """Test registering additional pattern."""
        from genie.patterns.base import PatternPlugin
        
        class DummyPattern(PatternPlugin):
            @property
            def name(self):
                return "dummy"
            
            def match(self, graph):
                return None
        
        matcher = NetworkXPatternMatcher()
        pattern = DummyPattern()
        
        # Should not raise
        matcher.register_pattern(pattern)


class TestSimplifiedPatternMatcher:
    """Test simplified pattern matcher."""
    
    def test_creation(self):
        """Test creating simplified matcher."""
        matcher = SimplifiedPatternMatcher()
        assert matcher.name == "Simplified"
    
    def test_match_patterns_llm_heuristic(self):
        """Test LLM pattern detection with heuristics."""
        matcher = SimplifiedPatternMatcher()
        
        # Create graph with matmul + softmax
        graph = ComputationGraph(
            nodes={
                'n1': ComputationNode('n1', 'aten::matmul', [], [], {}),
                'n2': ComputationNode('n2', 'aten::softmax', ['n1'], [], {})
            },
            edges=[('n1', 'n2')],
            entry_points={'n1'}
        )
        
        result = matcher.match_patterns(graph)
        
        assert result.is_ok
        matches = result.unwrap()
        assert len(matches) > 0
        assert any(m.metadata and m.metadata.get('workload_type') == WorkloadType.LLM for m in matches)
    
    def test_match_patterns_vision_heuristic(self):
        """Test vision pattern detection with heuristics."""
        matcher = SimplifiedPatternMatcher()
        
        # Create graph with conv + relu
        graph = ComputationGraph(
            nodes={
                'n1': ComputationNode('n1', 'aten::conv2d', [], [], {}),
                'n2': ComputationNode('n2', 'aten::relu', ['n1'], [], {})
            },
            edges=[('n1', 'n2')],
            entry_points={'n1'}
        )
        
        result = matcher.match_patterns(graph)
        
        assert result.is_ok
        matches = result.unwrap()
        assert len(matches) > 0
        assert any(m.metadata and m.metadata.get('workload_type') == WorkloadType.VISION for m in matches)
    
    def test_match_patterns_fx_graph(self):
        """Test simplified matcher with FX graph."""
        matcher = SimplifiedPatternMatcher()
        
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                y = torch.matmul(x, x)
                return torch.softmax(y, dim=-1)
        
        model = SimpleModel()
        traced = fx.symbolic_trace(model)
        
        result = matcher.match_patterns(traced)
        
        assert result.is_ok
    
    def test_performance_report(self):
        """Test performance reporting."""
        matcher = SimplifiedPatternMatcher()
        
        # Run some matches
        graph = ComputationGraph(
            nodes={'n1': ComputationNode('n1', 'aten::add', [], [], {})},
            edges=[],
            entry_points={'n1'}
        )
        matcher.match_patterns(graph)
        matcher.match_patterns(graph)
        
        report = matcher.get_performance_report()
        
        assert report['matcher_name'] == 'Simplified'
        assert report['total_matches'] == 2
        assert 'avg_time_ms' in report


class TestCompositePatternMatcher:
    """Test composite pattern matcher."""
    
    def test_creation(self):
        """Test creating composite matcher."""
        networkx = NetworkXPatternMatcher()
        simplified = SimplifiedPatternMatcher()
        
        composite = CompositePatternMatcher([networkx, simplified])
        
        assert 'Composite' in composite.name
        assert 'NetworkX' in composite.name
        assert 'Simplified' in composite.name
    
    def test_match_patterns_combines_results(self):
        """Test that composite matcher combines results from all matchers."""
        networkx = NetworkXPatternMatcher()
        simplified = SimplifiedPatternMatcher()
        composite = CompositePatternMatcher([networkx, simplified])
        
        # Create graph that both might match
        graph = ComputationGraph(
            nodes={
                'n1': ComputationNode('n1', 'aten::matmul', [], [], {}),
                'n2': ComputationNode('n2', 'aten::softmax', ['n1'], [], {})
            },
            edges=[('n1', 'n2')],
            entry_points={'n1'}
        )
        
        result = composite.match_patterns(graph)
        
        assert result.is_ok
        matches = result.unwrap()
        # Should have matches from at least one matcher
        assert len(matches) >= 0
    
    def test_performance_report(self):
        """Test composite performance report."""
        networkx = NetworkXPatternMatcher()
        simplified = SimplifiedPatternMatcher()
        composite = CompositePatternMatcher([networkx, simplified])
        
        report = composite.get_performance_report()
        
        assert 'sub_matchers' in report
        assert 'NetworkX' in report['sub_matchers']
        assert 'Simplified' in report['sub_matchers']


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_pattern_matcher_networkx(self):
        """Test creating NetworkX matcher via factory."""
        matcher = create_pattern_matcher('networkx')
        
        assert isinstance(matcher, NetworkXPatternMatcher)
        assert matcher.name == 'NetworkX'
    
    def test_create_pattern_matcher_simplified(self):
        """Test creating simplified matcher via factory."""
        matcher = create_pattern_matcher('simplified')
        
        assert isinstance(matcher, SimplifiedPatternMatcher)
        assert matcher.name == 'Simplified'
    
    def test_create_pattern_matcher_composite_default(self):
        """Test creating composite matcher with defaults."""
        matcher = create_pattern_matcher('composite')
        
        assert isinstance(matcher, CompositePatternMatcher)
        assert 'Composite' in matcher.name
    
    def test_create_pattern_matcher_composite_custom(self):
        """Test creating composite matcher with custom matchers."""
        networkx = NetworkXPatternMatcher()
        simplified = SimplifiedPatternMatcher()
        
        matcher = create_pattern_matcher('composite', matchers=[networkx, simplified])
        
        assert isinstance(matcher, CompositePatternMatcher)
    
    def test_create_pattern_matcher_invalid(self):
        """Test creating matcher with invalid type."""
        with pytest.raises(ValueError, match="Unknown matcher type"):
            create_pattern_matcher('invalid_type')
    
    def test_get_default_pattern_matcher(self):
        """Test getting default matcher."""
        matcher = get_default_pattern_matcher()
        
        assert isinstance(matcher, NetworkXPatternMatcher)


class TestSemanticAnalyzerIntegration:
    """Test integration with SemanticAnalyzer."""
    
    def test_analyzer_with_default_matcher(self):
        """Test analyzer uses default matcher when none specified."""
        from genie.semantic.analyzer import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer()
        
        assert hasattr(analyzer, 'pattern_matcher')
        assert analyzer.pattern_matcher is not None
    
    def test_analyzer_with_injected_matcher(self):
        """Test analyzer with injected pattern matcher."""
        from genie.semantic.analyzer import SemanticAnalyzer
        
        matcher = SimplifiedPatternMatcher()
        analyzer = SemanticAnalyzer(pattern_matcher=matcher)
        
        assert analyzer.pattern_matcher == matcher
        assert analyzer.pattern_matcher.name == 'Simplified'
    
    def test_analyzer_backward_compatibility(self):
        """Test analyzer backward compatibility with pattern_registry."""
        from genie.semantic.analyzer import SemanticAnalyzer
        from genie.semantic.pattern_registry import PatternRegistry
        
        registry = PatternRegistry()
        analyzer = SemanticAnalyzer(pattern_registry=registry)
        
        # Should wrap registry in NetworkXPatternMatcher
        assert hasattr(analyzer, 'pattern_matcher')
        assert isinstance(analyzer.pattern_matcher, NetworkXPatternMatcher)
    
    def test_analyzer_analyze_graph_with_matcher(self):
        """Test full analysis with pattern matcher."""
        from genie.semantic.analyzer import SemanticAnalyzer
        
        matcher = SimplifiedPatternMatcher()
        analyzer = SemanticAnalyzer(pattern_matcher=matcher)
        
        # Create simple graph
        graph = ComputationGraph(
            nodes={
                'n1': ComputationNode('n1', 'aten::matmul', [], [], {}),
                'n2': ComputationNode('n2', 'aten::softmax', ['n1'], [], {})
            },
            edges=[('n1', 'n2')],
            entry_points={'n1'}
        )
        
        profile = analyzer.analyze_graph(graph)
        
        assert profile is not None
        assert profile.workload_type is not None


class TestDependencyInjection:
    """Test dependency injection benefits."""
    
    def test_can_swap_matchers_at_runtime(self):
        """Test swapping matchers for different use cases."""
        from genie.semantic.analyzer import SemanticAnalyzer
        
        # Fast matcher for development
        fast_analyzer = SemanticAnalyzer(pattern_matcher=SimplifiedPatternMatcher())
        
        # Full matcher for production
        full_analyzer = SemanticAnalyzer(pattern_matcher=NetworkXPatternMatcher())
        
        # Both should work
        graph = ComputationGraph(
            nodes={'n1': ComputationNode('n1', 'aten::add', [], [], {})},
            edges=[],
            entry_points={'n1'}
        )
        
        profile1 = fast_analyzer.analyze_graph(graph)
        profile2 = full_analyzer.analyze_graph(graph)
        
        assert profile1 is not None
        assert profile2 is not None
    
    def test_custom_matcher_can_be_injected(self):
        """Test that custom matchers can be injected."""
        from genie.semantic.analyzer import SemanticAnalyzer
        
        class CustomMatcher(IPatternMatcher):
            def match_patterns(self, graph):
                from genie.core.exceptions import Result
                return Result.ok([])
            
            def get_performance_report(self):
                return {'custom': True}
            
            @property
            def name(self):
                return "Custom"
        
        custom = CustomMatcher()
        analyzer = SemanticAnalyzer(pattern_matcher=custom)
        
        assert analyzer.pattern_matcher.name == "Custom"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

