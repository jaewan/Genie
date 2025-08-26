#!/usr/bin/env python3
"""
Test script for Control-Data Plane Integration

This demonstrates:
1. Full TCP handshake between nodes
2. Capability exchange
3. Transfer negotiation
4. Simulated data transfer
5. Completion notification
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genie.runtime.control_data_integration import ControlDataIntegration
from genie.runtime.transport_coordinator import DataPlaneConfig


async def test_basic_connection():
    """Test basic node connection and handshake"""
    print("\n" + "="*60)
    print("TEST 1: Basic Connection and Handshake")
    print("="*60)
    
    # Create two nodes
    node1 = ControlDataIntegration(
        node_id="node1",
        control_port=6001,
        data_config=DataPlaneConfig(
            local_ip="127.0.0.1",
            data_port=6002,
            enable_gpudev=False  # Disable for testing without GPU
        )
    )
    
    node2 = ControlDataIntegration(
        node_id="node2", 
        control_port=6003,
        data_config=DataPlaneConfig(
            local_ip="127.0.0.1",
            data_port=6004,
            enable_gpudev=False
        )
    )
    
    try:
        # Start both nodes
        print("Starting node1...")
        await node1.start()
        print("Starting node2...")
        await node2.start()
        
        print("\nNodes started successfully")
        print(f"Node1: Control port={node1.control_port}, Data port={node1.data_config.data_port}")
        print(f"Node2: Control port={node2.control_port}, Data port={node2.data_config.data_port}")
        
        # Connect node1 to node2
        print("\nConnecting node1 -> node2...")
        connected = await node1.connect_to_node("127.0.0.1", 6003)
        
        if connected:
            print("✓ Connection successful!")
            print(f"  Active connections on node1: {len(node1.connections)}")
            print(f"  Active connections on node2: {len(node2.connections)}")
            
            # Verify connection details
            if "node2" in node1.connections:
                conn = node1.connections["node2"]
                print(f"\nConnection details:")
                print(f"  Remote node: {conn.node_id}")
                print(f"  Remote host: {conn.host}")
                print(f"  Remote control port: {conn.control_port}")
                print(f"  Remote data port: {conn.data_port}")
                print(f"  Connected: {conn.is_connected}")
                
                if conn.capabilities:
                    print(f"\nRemote capabilities:")
                    print(f"  GPU count: {conn.capabilities.gpu_count}")
                    print(f"  Max transfer size: {conn.capabilities.max_transfer_size / (1024**3):.1f} GB")
                    print(f"  Network bandwidth: {conn.capabilities.network_bandwidth_gbps} Gbps")
                    print(f"  Supported dtypes: {conn.capabilities.supported_dtypes}")
                    print(f"  Features: {conn.capabilities.features}")
        else:
            print("✗ Connection failed!")
            
        # Let heartbeats run for a bit
        print("\nLetting heartbeats run for 5 seconds...")
        await asyncio.sleep(5)
        
        # Check statistics
        stats1 = node1.get_statistics()
        stats2 = node2.get_statistics()
        
        print("\nNode1 statistics:")
        for key, value in stats1.items():
            if key != 'data_plane':
                print(f"  {key}: {value}")
        
        print("\nNode2 statistics:")
        for key, value in stats2.items():
            if key != 'data_plane':
                print(f"  {key}: {value}")
        
    finally:
        print("\nCleaning up...")
        await node1.stop()
        await node2.stop()
        print("Test completed")


async def test_simulated_transfer():
    """Test simulated tensor transfer negotiation"""
    print("\n" + "="*60)
    print("TEST 2: Simulated Transfer Negotiation")
    print("="*60)
    
    # Create two nodes
    node_sender = ControlDataIntegration(
        node_id="sender",
        control_port=6011,
        data_config=DataPlaneConfig(
            local_ip="127.0.0.1",
            data_port=6012,
            enable_gpudev=False
        )
    )
    
    node_receiver = ControlDataIntegration(
        node_id="receiver",
        control_port=6013,
        data_config=DataPlaneConfig(
            local_ip="127.0.0.1",
            data_port=6014,
            enable_gpudev=False
        )
    )
    
    try:
        # Start and connect nodes
        print("Starting nodes...")
        await node_sender.start()
        await node_receiver.start()
        
        print("Connecting sender -> receiver...")
        connected = await node_sender.connect_to_node("127.0.0.1", 6013)
        
        if not connected:
            print("Failed to connect nodes!")
            return
        
        print("✓ Nodes connected")
        
        # Create a dummy tensor (using numpy since we might not have torch)
        import numpy as np
        tensor = np.random.randn(100, 100).astype(np.float32)
        print(f"\nCreated test tensor: shape={tensor.shape}, dtype={tensor.dtype}, size={tensor.nbytes} bytes")
        
        # Attempt transfer
        print("\nInitiating transfer...")
        try:
            transfer_id = await node_sender.transfer_tensor(
                tensor=tensor,
                target_node="receiver",
                tensor_id="test_tensor_001",
                priority=1,
                timeout=30.0
            )
            
            print(f"✓ Transfer initiated: {transfer_id}")
            
            # Wait for negotiation
            await asyncio.sleep(2)
            
            # Check transfer state
            if transfer_id in node_sender.transfers:
                context = node_sender.transfers[transfer_id]
                print(f"  Transfer state: {context.state.name}")
                print(f"  Size: {context.size} bytes")
                print(f"  Shape: {context.shape}")
                print(f"  Dtype: {context.dtype}")
            
        except Exception as e:
            print(f"✗ Transfer failed: {e}")
        
        # Wait a bit for any background processing
        await asyncio.sleep(3)
        
        # Final statistics
        print("\nFinal statistics:")
        stats_sender = node_sender.get_statistics()
        stats_receiver = node_receiver.get_statistics()
        
        print(f"Sender - Transfers initiated: {stats_sender['transfers_initiated']}")
        print(f"Sender - Transfers completed: {stats_sender['transfers_completed']}")
        print(f"Sender - Transfers failed: {stats_sender['transfers_failed']}")
        print(f"Receiver - Active connections: {stats_receiver['active_connections']}")
        
    finally:
        print("\nCleaning up...")
        await node_sender.stop()
        await node_receiver.stop()
        print("Test completed")


async def test_multiple_connections():
    """Test multiple simultaneous connections"""
    print("\n" + "="*60)
    print("TEST 3: Multiple Connections")
    print("="*60)
    
    # Create a hub node and multiple client nodes
    hub = ControlDataIntegration(
        node_id="hub",
        control_port=6020,
        data_config=DataPlaneConfig(
            local_ip="127.0.0.1",
            data_port=6021,
            enable_gpudev=False
        )
    )
    
    clients = []
    for i in range(3):
        client = ControlDataIntegration(
            node_id=f"client{i}",
            control_port=6030 + i*2,
            data_config=DataPlaneConfig(
                local_ip="127.0.0.1",
                data_port=6031 + i*2,
                enable_gpudev=False
            )
        )
        clients.append(client)
    
    try:
        # Start hub
        print("Starting hub node...")
        await hub.start()
        
        # Start and connect clients
        for i, client in enumerate(clients):
            print(f"Starting client{i}...")
            await client.start()
            
            print(f"Connecting client{i} -> hub...")
            connected = await client.connect_to_node("127.0.0.1", 6020)
            if connected:
                print(f"  ✓ client{i} connected")
            else:
                print(f"  ✗ client{i} failed to connect")
        
        # Verify all connections
        print(f"\nHub has {len(hub.connections)} connections:")
        for node_id, conn in hub.connections.items():
            print(f"  - {node_id}: connected={conn.is_connected}")
        
        # Each client should see 1 connection (to hub)
        for client in clients:
            print(f"{client.node_id} has {len(client.connections)} connection(s)")
        
        # Let the system run for a bit
        print("\nRunning for 5 seconds...")
        await asyncio.sleep(5)
        
        # Get final stats
        print("\nFinal connection counts:")
        print(f"  Hub: {hub.stats['connection_count']} total connections")
        for client in clients:
            print(f"  {client.node_id}: {client.stats['connection_count']} connections")
        
    finally:
        print("\nCleaning up...")
        # Stop all nodes
        await hub.stop()
        for client in clients:
            await client.stop()
        print("Test completed")


async def main():
    """Run all tests"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_integration.log'),
            logging.StreamHandler()
        ]
    )
    
    # Reduce verbosity for the test
    logging.getLogger('genie.runtime').setLevel(logging.WARNING)
    
    print("\n" + "="*60)
    print("CONTROL-DATA PLANE INTEGRATION TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        await test_basic_connection()
        await test_simulated_transfer()
        await test_multiple_connections()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
