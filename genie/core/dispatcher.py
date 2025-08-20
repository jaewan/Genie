"""
Enhanced dispatcher integration with improved operation coverage and reliability.

This module provides a practical enhancement to the original dispatcher while
maintaining full backward compatibility.
"""
import logging
from typing import Any, Dict

# Import the enhanced dispatcher
from .enhanced_dispatcher import (
    enhanced_dispatcher,
    create_lazy_tensor_for_device_op,
    set_enhanced_lazy_mode,
    get_enhanced_stats
)

logger = logging.getLogger(__name__)


class DispatcherIntegration:
    """Backward compatibility wrapper for the enhanced dispatcher."""
    
    def __init__(self):
        self._enhanced = enhanced_dispatcher
    
    def set_lazy_mode(self, enabled: bool) -> None:
        """Enable or disable lazy execution mode."""
        self._enhanced.set_lazy_mode(enabled)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics in legacy format."""
        enhanced_stats = self._enhanced.get_stats()
        # Convert to old format for compatibility
        return {
            "registered_ops": enhanced_stats.get("registered_ops", 0),
            "fallback_ops": enhanced_stats.get("failed_registrations", 0),
            "operation_count": enhanced_stats.get("operation_count", 0),
            "lazy_mode": enhanced_stats.get("lazy_mode", True)
        }
    
    def register_op(self, op_name: str):
        """Deprecated: Use enhanced dispatcher registration instead."""
        import warnings
        warnings.warn(f"register_op is deprecated for {op_name}. Operations are now pre-registered in enhanced_dispatcher.py", UserWarning)
        
        def decorator(func):
            warnings.warn(f"Skipping manual registration of {op_name}, using enhanced approach", UserWarning)
            return func
        return decorator


# Create backward-compatible dispatcher instance
dispatcher = DispatcherIntegration()

# Expose the enhanced dispatcher for advanced usage
enhanced = enhanced_dispatcher

# Utility functions
def set_lazy_mode(enabled: bool) -> None:
    """Set lazy mode for the dispatcher."""
    set_enhanced_lazy_mode(enabled)

def get_dispatcher_stats() -> Dict[str, Any]:
    """Get comprehensive dispatcher statistics."""
    return get_enhanced_stats()

logger.info("Enhanced dispatcher module loaded successfully")


