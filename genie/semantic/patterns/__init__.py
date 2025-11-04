"""Pattern matching for computation graphs."""

from .base_matcher import PatternMatcher, Pattern, PatternRegistry, get_pattern_registry
from .attention_matcher import AttentionMatcher, ConvolutionMatcher, KVCacheMatcher

# Register default pattern matchers in the global registry
def _register_default_matchers():
    """Register the default pattern matchers."""
    registry = get_pattern_registry()

    # Create and register matchers
    attention_matcher = AttentionMatcher()
    convolution_matcher = ConvolutionMatcher()
    kv_cache_matcher = KVCacheMatcher()

    registry.register(attention_matcher)
    registry.register(convolution_matcher)
    registry.register(kv_cache_matcher)

# Register default matchers when module is imported
_register_default_matchers()

__all__ = [
    'PatternMatcher',
    'Pattern',
    'PatternRegistry',
    'get_pattern_registry',
    'AttentionMatcher',
    'ConvolutionMatcher',
    'KVCacheMatcher',
]
