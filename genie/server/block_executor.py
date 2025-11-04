"""
Phase 3: Remote Block Execution

Server-side execution of TorchScript blocks with pipelining support.

NOTE: RemoteBlockExecutor and PipelinedBlockExecutor implementations have been removed
as they are Phase 3+ experimental features not used in Phase 1/2 production code.
They can be re-added from git history when Phase 3 development begins.
If needed for reference, see git log for previous implementation.
"""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a single block execution."""
    block_id: int
    block_name: str
    outputs: Dict[str, torch.Tensor]
    execution_time_ms: float
    status: str = 'success'
    error: Optional[str] = None
