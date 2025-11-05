"""
Unit tests for SRGFusionCompiler.

Tests cover:
- Pattern detection (attention blocks, conv blocks)
- Grouping by execution phase
- Statistics tracking
- Edge cases and error handling
"""

import pytest
from typing import List, Dict, Any

from djinn.server.fusion_compiler import (
    SRGFusionCompiler,
    FusedBlock,
    FusionStrategy
)
from djinn.core.types import ExecutionPhase


@pytest.fixture
def compiler():
    """Create a fusion compiler for testing."""
    return SRGFusionCompiler(enable_torchscript=False, enable_compilation=False)


def create_operation(
    op_id: str,
    operation: str,
    phase: ExecutionPhase = ExecutionPhase.LLM_DECODE
) -> Dict[str, Any]:
    """Helper to create test operations."""
    return {
        'op_id': op_id,
        'operation': operation,
        'inputs': [],
        'kwargs': {}
    }


def create_metadata(
    op_id: str,
    phase: ExecutionPhase = ExecutionPhase.LLM_DECODE
) -> Dict[str, Any]:
    """Helper to create test metadata."""
    return {
        op_id: {
            'execution_phase': phase,
            'operation_type': 'operation'
        }
    }


class TestAttentionPatternDetection:
    """Test attention pattern detection."""

    def test_detect_attention_pattern(self, compiler):
        """Test detection of basic attention pattern."""
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::softmax'),
            create_operation('op3', 'aten::matmul'),
            create_operation('op4', 'aten::dropout'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.LLM_DECODE))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # Should group into 1 attention block
        assert len(blocks) == 1
        assert blocks[0].fusion_strategy == FusionStrategy.ATTENTION
        assert len(blocks[0].operations) == 4

    def test_multiple_attention_blocks(self, compiler):
        """Test detection of multiple attention patterns."""
        # First attention
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::softmax'),
            create_operation('op3', 'aten::matmul'),
            create_operation('op4', 'aten::linear'),  # Breaks pattern
            # Second attention
            create_operation('op5', 'aten::matmul'),
            create_operation('op6', 'aten::softmax'),
            create_operation('op7', 'aten::matmul'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.LLM_DECODE))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # Should have: attention + linear + attention
        assert len(blocks) == 3
        assert blocks[0].fusion_strategy == FusionStrategy.ATTENTION
        assert blocks[1].fusion_strategy == FusionStrategy.NONE
        assert blocks[2].fusion_strategy == FusionStrategy.ATTENTION

    def test_attention_with_residual(self, compiler):
        """Test attention pattern with residual connection."""
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::softmax'),
            create_operation('op3', 'aten::matmul'),
            create_operation('op4', 'aten::add'),  # Residual
            create_operation('op5', 'aten::layer_norm'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.LLM_DECODE))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # All should be grouped as attention (add and layer_norm are continuations)
        assert len(blocks) == 1
        assert blocks[0].fusion_strategy == FusionStrategy.ATTENTION
        assert len(blocks[0].operations) == 5


class TestConvPatternDetection:
    """Test convolution pattern detection."""

    def test_detect_conv_pattern(self, compiler):
        """Test detection of conv + batchnorm + activation pattern."""
        operations = [
            create_operation('op1', 'aten::convolution'),
            create_operation('op2', 'aten::batch_norm'),
            create_operation('op3', 'aten::relu'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.VISION_ENCODING))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # Should group into 1 conv block
        assert len(blocks) == 1
        assert blocks[0].fusion_strategy == FusionStrategy.CONV
        assert len(blocks[0].operations) == 3

    def test_conv_with_pooling(self, compiler):
        """Test conv pattern with pooling."""
        operations = [
            create_operation('op1', 'aten::convolution'),
            create_operation('op2', 'aten::batch_norm'),
            create_operation('op3', 'aten::relu'),
            create_operation('op4', 'aten::max_pool2d'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.VISION_ENCODING))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # All should be grouped as conv
        assert len(blocks) == 1
        assert blocks[0].fusion_strategy == FusionStrategy.CONV
        assert len(blocks[0].operations) == 4

    def test_multiple_conv_blocks(self, compiler):
        """Test detection of multiple conv blocks."""
        operations = [
            create_operation('op1', 'aten::convolution'),
            create_operation('op2', 'aten::relu'),
            create_operation('op3', 'aten::max_pool2d'),
            create_operation('op4', 'aten::flatten'),  # Breaks pattern
            create_operation('op5', 'aten::convolution'),
            create_operation('op6', 'aten::gelu'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.VISION_ENCODING))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # Should have: conv + flatten + conv
        assert len(blocks) == 3
        assert blocks[0].fusion_strategy == FusionStrategy.CONV
        assert blocks[1].fusion_strategy == FusionStrategy.NONE
        assert blocks[2].fusion_strategy == FusionStrategy.CONV


class TestPhaseGrouping:
    """Test grouping operations by execution phase."""

    def test_group_by_phase(self, compiler):
        """Test that operations are grouped by phase."""
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::relu'),
            create_operation('op3', 'aten::matmul'),
        ]
        
        metadata = {
            'op1': {'execution_phase': ExecutionPhase.LLM_DECODE},
            'op2': {'execution_phase': ExecutionPhase.VISION_ENCODING},
            'op3': {'execution_phase': ExecutionPhase.LLM_DECODE},
        }
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # op2 should be in its own phase
        assert len(blocks) >= 2
        
        # Verify phases are separated
        phases = [block.execution_phase for block in blocks]
        assert ExecutionPhase.LLM_DECODE in phases
        assert ExecutionPhase.VISION_ENCODING in phases

    def test_unknown_phase_handling(self, compiler):
        """Test handling of unknown execution phases."""
        operations = [
            create_operation('op1', 'aten::relu'),
            create_operation('op2', 'aten::sigmoid'),
        ]
        
        metadata = {
            'op1': {'execution_phase': ExecutionPhase.UNKNOWN},
            'op2': {'execution_phase': ExecutionPhase.UNKNOWN},
        }
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # Should still create blocks for unknown phases
        assert len(blocks) == 2
        assert all(block.execution_phase == ExecutionPhase.UNKNOWN for block in blocks)


class TestStatisticsTracking:
    """Test statistics tracking in fusion compiler."""

    def test_attention_blocks_counted(self, compiler):
        """Test that attention blocks are counted in statistics."""
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::softmax'),
            create_operation('op3', 'aten::matmul'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.LLM_DECODE))
        
        compiler.fuse_subgraph(operations, metadata)
        
        stats = compiler.get_stats()
        assert stats['attention_blocks_identified'] == 1

    def test_conv_blocks_counted(self, compiler):
        """Test that conv blocks are counted in statistics."""
        operations = [
            create_operation('op1', 'aten::convolution'),
            create_operation('op2', 'aten::relu'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.VISION_ENCODING))
        
        compiler.fuse_subgraph(operations, metadata)
        
        stats = compiler.get_stats()
        assert stats['conv_blocks_identified'] == 1

    def test_no_fusion_counted(self, compiler):
        """Test that non-fusable operations are counted."""
        operations = [
            create_operation('op1', 'aten::relu'),
            create_operation('op2', 'aten::sigmoid'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.UNKNOWN))
        
        compiler.fuse_subgraph(operations, metadata)
        
        stats = compiler.get_stats()
        assert stats['no_fusion_blocks'] == 2


class TestFusedBlockProperties:
    """Test properties of fused blocks."""

    def test_fused_block_has_correct_properties(self, compiler):
        """Test that fused blocks have all required properties."""
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::softmax'),
            create_operation('op3', 'aten::matmul'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.LLM_DECODE))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        block = blocks[0]
        assert block.block_id is not None
        assert block.operations == operations
        assert block.execution_phase == ExecutionPhase.LLM_DECODE
        assert block.fusion_strategy == FusionStrategy.ATTENTION
        assert block.compiled_module is None
        assert block.torchscript_compiled is False

    def test_fused_block_operations_preserved(self, compiler):
        """Test that original operations are preserved in fused block."""
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::softmax'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.LLM_DECODE))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # Original operations should be in block
        assert len(blocks[0].operations) == 2
        assert blocks[0].operations[0]['op_id'] == 'op1'
        assert blocks[0].operations[1]['op_id'] == 'op2'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_operations_list(self, compiler):
        """Test handling of empty operations list."""
        blocks = compiler.fuse_subgraph([], {})
        
        assert len(blocks) == 0

    def test_single_operation(self, compiler):
        """Test handling of single operation."""
        operations = [create_operation('op1', 'aten::relu')]
        metadata = {'op1': {'execution_phase': ExecutionPhase.UNKNOWN}}
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        assert len(blocks) == 1
        assert blocks[0].fusion_strategy == FusionStrategy.NONE

    def test_missing_metadata(self, compiler):
        """Test handling of missing metadata for operations."""
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::softmax'),
        ]
        
        metadata = {
            'op1': {'execution_phase': ExecutionPhase.LLM_DECODE}
            # op2 metadata is missing
        }
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # Should still process, using default phase for missing metadata
        assert len(blocks) >= 1

    def test_various_matmul_variants(self, compiler):
        """Test detection of various matmul variants."""
        operations = [
            create_operation('op1', 'aten::matmul'),
            create_operation('op2', 'aten::bmm'),
            create_operation('op3', 'aten::mm'),
            create_operation('op4', 'aten::addmm'),
        ]
        
        metadata = {}
        for op in operations:
            metadata.update(create_metadata(op['op_id'], ExecutionPhase.LLM_DECODE))
        
        blocks = compiler.fuse_subgraph(operations, metadata)
        
        # All matmul variants should be grouped as attention
        assert len(blocks) == 1
        assert blocks[0].fusion_strategy == FusionStrategy.ATTENTION


class TestCompilerStatistics:
    """Test overall compiler statistics."""

    def test_fusion_time_tracked(self, compiler):
        """Test that fusion time is tracked."""
        operations = [create_operation('op1', 'aten::relu')] * 100
        metadata = {f'op{i}': {'execution_phase': ExecutionPhase.UNKNOWN} for i in range(100)}
        
        compiler.fuse_subgraph(operations, metadata)
        
        stats = compiler.get_stats()
        assert stats['fusion_time_ms'] >= 0

    def test_statistics_accumulate(self, compiler):
        """Test that statistics accumulate over multiple calls."""
        for i in range(3):
            operations = [
                create_operation('op1', 'aten::matmul'),
                create_operation('op2', 'aten::softmax'),
            ]
            
            metadata = {
                'op1': {'execution_phase': ExecutionPhase.LLM_DECODE},
                'op2': {'execution_phase': ExecutionPhase.LLM_DECODE},
            }
            
            compiler.fuse_subgraph(operations, metadata)
        
        stats = compiler.get_stats()
        # Should have identified 3 attention blocks across all calls
        assert stats['attention_blocks_identified'] == 3

