"""
Pipelining Integration for Genie Executor

Extends executor with network pipelining support while maintaining
backward compatibility with single-operation execution path.

This module provides:
- Feature flag for enabling/disabling pipelining
- Integration with PipelinedExecutor
- Fallback to sequential execution
- Configuration management
"""

import asyncio
import logging
from typing import Optional, Any
import torch

from genie.core.pipelined_executor import PipelinedExecutor
from genie.core.coordinator import get_coordinator

logger = logging.getLogger(__name__)


class PipeliningConfig:
    """Configuration for executor pipelining"""
    
    def __init__(
        self,
        enable_pipelining: bool = True,
        max_concurrent: int = 5,
        operation_timeout: float = 30.0,
        target_server: str = "localhost:5556"
    ):
        """
        Initialize pipelining configuration.
        
        Args:
            enable_pipelining: Enable/disable pipelining (default: True)
            max_concurrent: Max operations in flight (default: 5)
            operation_timeout: Timeout per operation in seconds (default: 30.0)
            target_server: Default remote server address
        """
        self.enable_pipelining = enable_pipelining
        self.max_concurrent = max_concurrent
        self.operation_timeout = operation_timeout
        self.target_server = target_server
    
    @staticmethod
    def for_workload(workload_type: str) -> 'PipeliningConfig':
        """Get optimal config for specific workload type"""
        
        if workload_type == "decode":
            # Small, sequential operations
            return PipeliningConfig(
                enable_pipelining=True,
                max_concurrent=5,  # Small ops benefit from higher concurrency
                operation_timeout=30.0
            )
        elif workload_type == "vision":
            # Larger batch operations
            return PipeliningConfig(
                enable_pipelining=True,
                max_concurrent=3,  # Fewer concurrent large ops
                operation_timeout=60.0
            )
        else:
            # Conservative default
            return PipeliningConfig(
                enable_pipelining=True,
                max_concurrent=3,
                operation_timeout=30.0
            )


class ExecutorWithPipelining:
    """
    Wrapper that adds pipelining support to executor.
    
    Provides:
    - Automatic pipelining for remote operations
    - Feature flag for gradual rollout
    - Statistics collection
    - Fallback to sequential execution
    """
    
    def __init__(self, base_executor, config: Optional[PipeliningConfig] = None):
        """
        Initialize executor with pipelining support.
        
        Args:
            base_executor: The base executor (SimpleExecutor or CachedGraphExecutor)
            config: PipeliningConfig instance (default: create standard config)
        """
        self.base_executor = base_executor
        self.config = config or PipeliningConfig()
        
        # Pipelining infrastructure
        self.pipelined_executor: Optional[PipelinedExecutor] = None
        self._pipeline_running = False
        
        # Statistics
        self.stats = {
            "total_executions": 0,
            "pipelined_executions": 0,
            "sequential_executions": 0,
            "total_latency_ms": 0.0,
            "pipelined_latency_ms": 0.0,
            "sequential_latency_ms": 0.0,
        }
        
        logger.info(f"ExecutorWithPipelining initialized (pipelining={self.config.enable_pipelining})")
    
    async def start_pipelining(self):
        """Start background pipelining tasks"""
        if not self.config.enable_pipelining:
            logger.info("Pipelining disabled by config")
            return
        
        if self._pipeline_running:
            logger.warning("Pipeline already running")
            return
        
        try:
            coordinator = get_coordinator()
            self.pipelined_executor = PipelinedExecutor(
                coordinator,
                max_concurrent=self.config.max_concurrent
            )
            
            await self.pipelined_executor.start()
            self._pipeline_running = True
            logger.info("Pipelining started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start pipelining: {e}")
            self._pipeline_running = False
    
    async def stop_pipelining(self):
        """Stop background pipelining tasks"""
        if self.pipelined_executor and self._pipeline_running:
            await self.pipelined_executor.stop()
            self._pipeline_running = False
            logger.info("Pipelining stopped")
    
    async def execute_remote_pipelined(
        self,
        lazy_tensor: Any,
        target: Optional[str] = None
    ) -> torch.Tensor:
        """
        Execute remote operation with pipelining if enabled.
        
        Args:
            lazy_tensor: The lazy tensor operation
            target: Remote server address (default from config)
            
        Returns:
            Result tensor
        """
        if not self._pipeline_running or not self.config.enable_pipelining:
            # Fallback to sequential
            return await self._execute_remote_sequential(lazy_tensor, target)
        
        target = target or self.config.target_server
        
        try:
            # Use pipelined execution
            result = await self.pipelined_executor.execute_pipelined(lazy_tensor, target)
            self.stats["pipelined_executions"] += 1
            return result
            
        except Exception as e:
            logger.warning(f"Pipelined execution failed, falling back: {e}")
            # Fallback to sequential on error
            return await self._execute_remote_sequential(lazy_tensor, target)
    
    async def _execute_remote_sequential(
        self,
        lazy_tensor: Any,
        target: Optional[str] = None
    ) -> torch.Tensor:
        """
        Execute remote operation sequentially (old path).
        
        This is the fallback path when pipelining is disabled or fails.
        """
        target = target or self.config.target_server
        self.stats["sequential_executions"] += 1
        
        # Call coordinator directly for single operation
        coordinator = get_coordinator()
        
        # Materialize inputs
        if isinstance(lazy_tensor, torch.Tensor):
            inputs = [lazy_tensor]
        else:
            inputs = [lazy_tensor]
        
        # Execute remotely
        result = await coordinator.execute_remote_operation_async(
            operation=getattr(lazy_tensor, 'operation', 'forward'),
            inputs=inputs,
            target=target,
            timeout=self.config.operation_timeout
        )
        
        return result
    
    def get_pipelining_stats(self) -> dict:
        """Get pipelining statistics"""
        stats = self.stats.copy()
        
        if self.pipelined_executor:
            stats["pipeline_status"] = self.pipelined_executor.get_pipeline_stats()
        
        # Calculate averages
        total = stats["total_executions"]
        if total > 0:
            stats["pipelined_ratio"] = stats["pipelined_executions"] / total
            stats["sequential_ratio"] = stats["sequential_executions"] / total
            
            if stats["pipelined_executions"] > 0:
                stats["avg_pipelined_latency_ms"] = (
                    stats["pipelined_latency_ms"] / stats["pipelined_executions"]
                )
            
            if stats["sequential_executions"] > 0:
                stats["avg_sequential_latency_ms"] = (
                    stats["sequential_latency_ms"] / stats["sequential_executions"]
                )
        
        return stats
    
    def enable_pipelining(self, enable: bool = True):
        """Enable or disable pipelining (feature flag)"""
        self.config.enable_pipelining = enable
        logger.info(f"Pipelining {'enabled' if enable else 'disabled'}")
    
    def set_max_concurrent(self, max_concurrent: int):
        """Adjust max concurrent operations"""
        self.config.max_concurrent = max_concurrent
        if self.pipelined_executor:
            self.pipelined_executor.state.max_concurrent = max_concurrent
        logger.info(f"Max concurrent set to {max_concurrent}")


# Integration helper function
def add_pipelining_to_executor(executor, config: Optional[PipeliningConfig] = None) -> ExecutorWithPipelining:
    """
    Wrap an existing executor with pipelining support.
    
    Usage:
        executor = SimpleExecutor()
        executor = add_pipelining_to_executor(executor)
        asyncio.run(executor.start_pipelining())
    
    Args:
        executor: The base executor to wrap
        config: Optional pipelining configuration
        
    Returns:
        ExecutorWithPipelining instance
    """
    return ExecutorWithPipelining(executor, config)
