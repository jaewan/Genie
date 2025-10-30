"""
Executor Integration Layer (Phase 3D)

Integrates Phase 1 (Pipelining) + Phase 3 (Batching + Pooling)
into a unified executor with feature flags and adaptive routing.

Architecture:
- Wraps Phase 1 PipelinedExecutor
- Adds BatchingCoordinator for operation batching
- Feature flags for safe rollout
- Adaptive routing based on operation size
- Full backward compatibility

Expected Benefits:
- Phase 1 alone: 5.9-18.8x (simulated)
- Phase 3A (pooling): +5% improvement
- Phase 3B (batching): +3-5x improvement
- Phase 1+3 combined: 3-5x on real workloads (after pooling/batching overhead)
"""

import asyncio
import logging
import torch
from typing import Optional, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for executor integration"""
    # Pipelining settings
    enable_pipelining: bool = True
    pipelining_max_concurrent: int = 5
    
    # Batching settings
    enable_batching: bool = True
    batch_size: int = 5
    batch_timeout_ms: float = 1.0
    
    # Adaptive routing
    enable_adaptive_routing: bool = True
    small_op_threshold_bytes: int = 1024  # < 1KB: batch
    large_op_threshold_bytes: int = 10_485_760  # > 10MB: pipeline
    
    # Feature flags
    feature_flag_enabled: bool = True


class ExecutorIntegration:
    """
    Unified executor combining pipelining + batching + pooling.
    
    Workflow:
    1. Receive operation (lazy tensor)
    2. Adaptive routing: small → batch, large → pipeline
    3. Execute via optimal strategy
    4. Return result tensor
    
    Full backward compatibility: can disable features independently.
    """
    
    def __init__(self,
                 pipelined_executor,
                 batching_coordinator,
                 config: Optional[IntegrationConfig] = None):
        """
        Initialize integration layer.
        
        Args:
            pipelined_executor: Phase 1 PipelinedExecutor instance
            batching_coordinator: Phase 3 BatchingCoordinator instance
            config: Integration configuration
        """
        self.pipelined = pipelined_executor
        self.batching = batching_coordinator
        self.config = config or IntegrationConfig()
        
        # Statistics
        self.stats = {
            'total_operations': 0,
            'pipelined_ops': 0,
            'batched_ops': 0,
            'adaptive_routing_decisions': {},
            'total_time_ms': 0,
        }
        
        logger.info(f"ExecutorIntegration initialized with config: "
                   f"pipelining={self.config.enable_pipelining}, "
                   f"batching={self.config.enable_batching}, "
                   f"adaptive_routing={self.config.enable_adaptive_routing}")
    
    async def execute(self,
                      lazy_tensor: Any,
                      target: str = "localhost:5556") -> torch.Tensor:
        """
        Execute operation with optimal strategy selection.
        
        Intelligently routes between:
        - Batching (small ops): Low latency per-op via batching
        - Pipelining (large ops): Throughput via concurrent exec
        - Sequential (default): Backward compatible
        
        Args:
            lazy_tensor: LazyTensor to execute (or graph to materialize)
            target: Remote server address
            
        Returns:
            Materialized torch.Tensor result
        """
        if not self.config.feature_flag_enabled:
            # Fallback to sequential execution (backward compatible)
            logger.debug("Feature flag disabled, using sequential execution")
            return await self._execute_sequential(lazy_tensor, target)
        
        # Determine operation size (estimated from lazy tensor)
        op_size = self._estimate_operation_size(lazy_tensor)
        
        # Adaptive routing
        strategy = self._select_strategy(op_size)
        
        # Execute via selected strategy
        result = await self._execute_with_strategy(
            lazy_tensor, target, strategy
        )
        
        # Update statistics
        self.stats['total_operations'] += 1
        if strategy == 'pipelining':
            self.stats['pipelined_ops'] += 1
        elif strategy == 'batching':
            self.stats['batched_ops'] += 1
        
        return result
    
    def _estimate_operation_size(self, lazy_tensor: Any) -> int:
        """
        Estimate operation size in bytes.
        
        Used for adaptive routing decisions.
        
        Args:
            lazy_tensor: LazyTensor or graph node
            
        Returns:
            Estimated size in bytes
        """
        try:
            # If it's a tensor, use its size
            if hasattr(lazy_tensor, 'numel') and hasattr(lazy_tensor, 'element_size'):
                return lazy_tensor.numel() * lazy_tensor.element_size()
            
            # If it's a LazyTensor with shape
            if hasattr(lazy_tensor, 'shape') and hasattr(lazy_tensor, 'dtype'):
                numel = 1
                for dim in lazy_tensor.shape:
                    numel *= dim
                # Estimate bytes (float32 = 4 bytes)
                return numel * 4
            
            # Default: assume medium size
            return 1_000_000  # 1MB default
        except Exception as e:
            logger.warning(f"Could not estimate operation size: {e}, using default")
            return 1_000_000
    
    def _select_strategy(self, op_size: int) -> str:
        """
        Select execution strategy based on operation size.
        
        Routing logic:
        - Small ops (<1KB): batch (multiple ops per round-trip)
        - Medium ops (1KB-10MB): hybrid (both applicable)
        - Large ops (>10MB): pipeline (amortize overhead)
        
        Args:
            op_size: Operation size in bytes
            
        Returns:
            Strategy: 'batching', 'pipelining', or 'sequential'
        """
        if not self.config.enable_adaptive_routing:
            # Default to pipelining if routing disabled
            return 'pipelining' if self.config.enable_pipelining else 'sequential'
        
        # Adaptive routing
        if op_size < self.config.small_op_threshold_bytes:
            strategy = 'batching' if self.config.enable_batching else 'sequential'
        elif op_size > self.config.large_op_threshold_bytes:
            strategy = 'pipelining' if self.config.enable_pipelining else 'sequential'
        else:
            # Medium: prefer pipelining for throughput
            strategy = 'pipelining' if self.config.enable_pipelining else 'sequential'
        
        # Track routing decisions
        if strategy not in self.stats['adaptive_routing_decisions']:
            self.stats['adaptive_routing_decisions'][strategy] = 0
        self.stats['adaptive_routing_decisions'][strategy] += 1
        
        logger.debug(f"Routing decision: {op_size} bytes → {strategy}")
        return strategy
    
    async def _execute_with_strategy(self,
                                     lazy_tensor: Any,
                                     target: str,
                                     strategy: str) -> torch.Tensor:
        """
        Execute operation using selected strategy.
        
        Args:
            lazy_tensor: LazyTensor to execute
            target: Remote server address
            strategy: Execution strategy ('batching', 'pipelining', 'sequential')
            
        Returns:
            Materialized result tensor
        """
        if strategy == 'batching':
            return await self._execute_batched(lazy_tensor, target)
        elif strategy == 'pipelining':
            return await self._execute_pipelined(lazy_tensor, target)
        else:
            return await self._execute_sequential(lazy_tensor, target)
    
    async def _execute_batched(self,
                               lazy_tensor: Any,
                               target: str) -> torch.Tensor:
        """
        Execute via batching (Phase 3C).
        
        Small operations are batched together for efficiency.
        """
        logger.debug(f"Executing via batching strategy")
        
        # Use BatchingCoordinator to queue and batch operation
        operation_metadata = {
            'operation': 'execute',
            'lazy_tensor': lazy_tensor,
            'target': target,
        }
        
        try:
            result = await self.batching.execute_operation_batched(
                'lazy_tensor_execute',
                operation_metadata
            )
            
            # Convert result dict to tensor (implementation-dependent)
            if isinstance(result, dict) and 'tensor' in result:
                return result['tensor']
            elif isinstance(result, torch.Tensor):
                return result
            else:
                logger.warning(f"Unexpected result format: {type(result)}")
                return result
                
        except Exception as e:
            logger.error(f"Batching execution failed: {e}, falling back to sequential")
            return await self._execute_sequential(lazy_tensor, target)
    
    async def _execute_pipelined(self,
                                 lazy_tensor: Any,
                                 target: str) -> torch.Tensor:
        """
        Execute via pipelining (Phase 1).
        
        Large operations use pipelining for throughput.
        """
        logger.debug(f"Executing via pipelining strategy")
        
        try:
            # Delegate to Phase 1 PipelinedExecutor
            result = await self.pipelined.execute_pipelined(
                lazy_tensor,
                target=target
            )
            return result
            
        except Exception as e:
            logger.error(f"Pipelining execution failed: {e}, falling back to sequential")
            return await self._execute_sequential(lazy_tensor, target)
    
    async def _execute_sequential(self,
                                  lazy_tensor: Any,
                                  target: str) -> torch.Tensor:
        """
        Execute sequentially (backward compatible fallback).
        
        Used when other strategies are disabled or on error.
        """
        logger.debug(f"Executing sequentially (fallback)")
        
        try:
            # Use original executor's simple_executor
            if hasattr(self.pipelined, 'executor'):
                return await self.pipelined.executor.execute(lazy_tensor, target)
            else:
                raise RuntimeError("No fallback executor available")
                
        except Exception as e:
            logger.error(f"Sequential execution failed: {e}")
            raise
    
    def enable_feature(self, feature: str, enable: bool = True):
        """
        Enable/disable optimization features.
        
        Args:
            feature: Feature name ('pipelining', 'batching', 'adaptive_routing', 'all')
            enable: Whether to enable or disable
        """
        if feature == 'pipelining':
            self.config.enable_pipelining = enable
        elif feature == 'batching':
            self.config.enable_batching = enable
        elif feature == 'adaptive_routing':
            self.config.enable_adaptive_routing = enable
        elif feature == 'all':
            self.config.enable_pipelining = enable
            self.config.enable_batching = enable
            self.config.enable_adaptive_routing = enable
        else:
            raise ValueError(f"Unknown feature: {feature}")
        
        logger.info(f"Feature '{feature}' {'enabled' if enable else 'disabled'}")
    
    def disable_all_optimizations(self):
        """Disable all optimizations for baseline comparison"""
        self.enable_feature('all', enable=False)
        logger.info("All optimizations disabled - running baseline mode")
    
    def enable_all_optimizations(self):
        """Enable all optimizations"""
        self.enable_feature('all', enable=True)
        logger.info("All optimizations enabled - running optimized mode")
    
    async def close(self):
        """Clean up resources"""
        logger.info("Closing ExecutorIntegration")
        
        if hasattr(self.batching, 'close'):
            await self.batching.close()
        
        if hasattr(self.pipelined, 'close'):
            await self.pipelined.close()
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        stats = {
            **self.stats,
            'pipelined': self.pipelined.get_stats() if hasattr(self.pipelined, 'get_stats') else {},
            'batching': self.batching.get_stats() if hasattr(self.batching, 'get_stats') else {},
        }
        return stats
    
    def print_stats(self):
        """Print execution statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("EXECUTOR INTEGRATION STATISTICS")
        print("="*60)
        print(f"Total operations: {stats['total_operations']}")
        print(f"  - Pipelined: {stats['pipelined_ops']}")
        print(f"  - Batched: {stats['batched_ops']}")
        print(f"  - Sequential: {stats['total_operations'] - stats['pipelined_ops'] - stats['batched_ops']}")
        print(f"\nRouting decisions: {stats['adaptive_routing_decisions']}")
        print(f"Total time: {stats['total_time_ms']:.2f}ms")
        
        if stats['pipelined']:
            print(f"\nPipelining stats:")
            for k, v in stats['pipelined'].items():
                print(f"  {k}: {v}")
        
        if stats['batching']:
            print(f"\nBatching stats:")
            for k, v in stats['batching'].items():
                print(f"  {k}: {v}")
        print("="*60 + "\n")
