"""
Server components for Djinn disaggregated GPU cluster.

Usage:
    # Start server
    python -m djinn.backend.server.server --gpus 4 --port 5555

    # Or programmatically
    from djinn.backend.server import DjinnServer
    server = DjinnServer(config)
    await server.start()
"""

from .server import DjinnServer
from .capability_provider import CapabilityProvider

__all__ = ['DjinnServer', 'CapabilityProvider']
