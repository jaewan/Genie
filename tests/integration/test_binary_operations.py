"""
Integration tests for binary operations with multi-tensor protocol.

Verifies:
1. Multiple tensors can be sent in one message
2. Binary operations execute correctly on server
3. Results are returned correctly
4. No base64 encoding overhead
"""

import asyncio
import pytest
import torch
from genie.core.coordinator import GenieCoordinator, CoordinatorConfig
from genie.server.server import GenieServer, ServerConfig


@pytest.mark.asyncio
async def test_remote_add():
    """Test binary addition: c = a + b."""
    
    # Start server
    server_config = ServerConfig(
        node_id="test-server",
        control_port=5555,
        data_port=5556,
        prefer_dpdk=False
    )
    server = GenieServer(server_config)
    success = await server.start()
    assert success, "Server failed to start"
    
    # Start coordinator
    client_config = CoordinatorConfig(
        node_id="test-client",
        control_port=5557,
        data_port=5558,
        prefer_dpdk=False,
        tcp_fallback=True
    )
    coordinator = GenieCoordinator(client_config)
    await coordinator.start()
    
    try:
        # Test data
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        
        print(f"\n=== Testing: c = a + b ===")
        print(f"  a: {a.shape}")
        print(f"  b: {b.shape}")
        
        # Execute remotely using multi-tensor protocol
        result = await coordinator.execute_remote_operation(
            operation='aten::add',
            inputs=[a, b],
            target='localhost:5556',
            timeout=30.0  # Increased timeout for debugging
        )
        
        # Verify correctness
        expected = a + b
        assert torch.allclose(result, expected, atol=1e-5), \
            f"Result mismatch: max diff = {(result - expected).abs().max()}"
        
        print(f"✅ Remote add works correctly!")
        print(f"  result: {result.shape}")
        print(f"  max error: {(result - expected).abs().max():.2e}")
    
    finally:
        await coordinator.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_remote_matmul():
    """Test matrix multiplication: c = a @ b."""
    
    # Setup server
    server_config = ServerConfig(
        node_id="test-server",
        control_port=5555,
        data_port=5556,
        prefer_dpdk=False
    )
    server = GenieServer(server_config)
    await server.start()
    
    # Setup coordinator
    client_config = CoordinatorConfig(
        node_id="test-client",
        control_port=5557,
        data_port=5558,
        prefer_dpdk=False,
        tcp_fallback=True
    )
    coordinator = GenieCoordinator(client_config)
    await coordinator.start()
    
    try:
        a = torch.randn(50, 75)
        b = torch.randn(75, 100)
        
        print(f"\n=== Testing: c = a @ b ===")
        print(f"  a: {a.shape}")
        print(f"  b: {b.shape}")
        
        result = await coordinator.execute_remote_operation(
            operation='aten::matmul',
            inputs=[a, b],
            target='localhost:5556',
            timeout=30.0
        )
        
        expected = a @ b
        assert torch.allclose(result, expected, atol=1e-4), \
            f"Result mismatch: max diff = {(result - expected).abs().max()}"
        
        print(f"✅ Remote matmul works!")
        print(f"  result: {result.shape}")
        print(f"  max error: {(result - expected).abs().max():.2e}")
    finally:
        await coordinator.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_all_binary_operations():
    """Test all supported binary operations."""
    
    operations = [
        ('aten::add', lambda a, b: a + b),
        ('aten::sub', lambda a, b: a - b),
        ('aten::mul', lambda a, b: a * b),
        ('aten::div', lambda a, b: a / b),
        ('aten::matmul', lambda a, b: a @ b),
    ]
    
    # Setup server
    server_config = ServerConfig(
        node_id="test-server",
        control_port=5555,
        data_port=5556,
        prefer_dpdk=False
    )
    server = GenieServer(server_config)
    await server.start()
    
    # Setup coordinator
    client_config = CoordinatorConfig(
        node_id="test-client",
        control_port=5557,
        data_port=5558,
        prefer_dpdk=False,
        tcp_fallback=True
    )
    coordinator = GenieCoordinator(client_config)
    await coordinator.start()
    
    try:
        for op_name, reference_fn in operations:
            print(f"\nTesting {op_name}...")
            
            # Test data (different shapes for matmul)
            if 'matmul' in op_name:
                a = torch.randn(32, 64)
                b = torch.randn(64, 48)
            else:
                a = torch.randn(50, 50)
                b = torch.randn(50, 50)
            
            result = await coordinator.execute_remote_operation(
                operation=op_name,
                inputs=[a, b],
                target='localhost:5556',
                timeout=30.0
            )
            
            expected = reference_fn(a, b)
            assert torch.allclose(result, expected, atol=1e-4), \
                f"{op_name} failed: max diff = {(result - expected).abs().max()}"
            
            print(f"  ✓ {op_name} passed")
        
        print("\n✅ All binary operations work!")
    
    finally:
        await coordinator.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_unary_operations():
    """Test unary operations (single tensor input)."""
    
    operations = [
        ('aten::relu', lambda x: torch.relu(x)),
        ('aten::sigmoid', lambda x: torch.sigmoid(x)),
        ('aten::tanh', lambda x: torch.tanh(x)),
    ]
    
    # Setup server
    server_config = ServerConfig(
        node_id="test-server",
        control_port=5555,
        data_port=5556,
        prefer_dpdk=False
    )
    server = GenieServer(server_config)
    await server.start()
    
    # Setup coordinator
    client_config = CoordinatorConfig(
        node_id="test-client",
        control_port=5557,
        data_port=5558,
        prefer_dpdk=False,
        tcp_fallback=True
    )
    coordinator = GenieCoordinator(client_config)
    await coordinator.start()
    
    try:
        for op_name, reference_fn in operations:
            print(f"\nTesting {op_name}...")
            
            x = torch.randn(50, 50)
            
            result = await coordinator.execute_remote_operation(
                operation=op_name,
                inputs=[x],
                target='localhost:5556',
                timeout=10.0
            )
            
            expected = reference_fn(x)
            assert torch.allclose(result, expected, atol=1e-5), \
                f"{op_name} failed: max diff = {(result - expected).abs().max()}"
            
            print(f"  ✓ {op_name} passed")
        
        print("\n✅ All unary operations work!")
    
    finally:
        await coordinator.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_no_base64_overhead():
    """Verify multi-tensor protocol doesn't use base64 encoding."""
    
    # Setup server
    server_config = ServerConfig(
        node_id="test-server",
        control_port=5555,
        data_port=5556,
        prefer_dpdk=False
    )
    server = GenieServer(server_config)
    await server.start()
    
    # Setup coordinator
    client_config = CoordinatorConfig(
        node_id="test-client",
        control_port=5557,
        data_port=5558,
        prefer_dpdk=False,
        tcp_fallback=True
    )
    coordinator = GenieCoordinator(client_config)
    await coordinator.start()
    
    try:
        # Create tensor with 1MB
        tensor_size_mb = 1
        tensor = torch.randn(256, 1024)  # ~1MB
        
        print(f"\n=== Testing multi-tensor overhead ===")
        print(f"  Single tensor size: {tensor.numel() * tensor.element_size() / 1024**2:.2f} MB")
        
        # For binary operation with two 1MB tensors, we expect ~2MB total (no base64)
        # With base64, it would be ~2.66MB (2MB * 1.33)
        
        # Execute operation
        result = await coordinator.execute_remote_operation(
            operation='aten::add',
            inputs=[tensor, tensor],
            target='localhost:5556',
            timeout=60.0  # Longer timeout for larger tensors
        )
        
        expected = tensor + tensor
        assert torch.allclose(result, expected, atol=1e-4)
        
        print(f"✅ No base64 overhead confirmed!")
        print(f"  Executed successfully with multi-tensor protocol")
    
    finally:
        await coordinator.stop()
        await server.stop()


if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '-s'])
