"""
Unit and integration tests for KV Session Manager Phase 1 enhancements.

Tests session pinning, async operations, idle cleanup, and statistics.
"""

import asyncio
import time

import numpy as np
import pytest
import torch

from djinn.server.kv_session_manager import KVSessionManager, KVSession


@pytest.mark.asyncio
class TestKVSessionManager:
    """Test KV session manager functionality."""
    
    async def test_manager_initialization(self):
        """Test manager initializes correctly."""
        mgr = KVSessionManager(idle_timeout_seconds=30.0, cleanup_interval_seconds=5.0)
        assert mgr.idle_timeout_seconds == 30.0
        assert mgr.cleanup_interval_seconds == 5.0
        assert len(mgr._sessions) == 0
    
    async def test_session_creation(self):
        """Test creating new session."""
        mgr = KVSessionManager()
        
        session = await mgr.get_or_create("session_1", gpu_id=0)
        
        assert session.session_id == "session_1"
        assert session.gpu_id == 0
        assert session.step_count == 1
        assert mgr.stats["sessions_created"] == 1
    
    async def test_session_creation_with_initial_kv(self):
        """Test creating session with initial KV cache."""
        mgr = KVSessionManager()
        
        # Create dummy KV tensor
        kv_tensor = torch.randn(1, 4, 32, 64)
        
        session = await mgr.get_or_create("session_1", gpu_id=0, initial_kv=kv_tensor)
        
        assert session.session_id == "session_1"
        assert session.kv_cache is not None
        assert session.bytes_used > 0
    
    async def test_session_retrieval(self):
        """Test retrieving existing session."""
        mgr = KVSessionManager()
        
        # Create session
        session1 = await mgr.get_or_create("session_1", gpu_id=0)
        initial_step = session1.step_count
        
        # Retrieve same session
        session2 = await mgr.get_or_create("session_1", gpu_id=0)
        
        # Should be same object
        assert session1 is session2
        assert session2.step_count == initial_step + 1
    
    async def test_kv_update(self):
        """Test updating KV cache in session."""
        mgr = KVSessionManager()
        
        # Create initial session
        await mgr.get_or_create("session_1", gpu_id=0)
        
        # Update with new KV
        new_kv = torch.randn(1, 5, 32, 64)
        session = await mgr.update_kv("session_1", new_kv)
        
        assert session.kv_cache is not None
        assert session.step_count == 2
    
    async def test_session_close(self):
        """Test closing a session."""
        mgr = KVSessionManager()
        
        kv_tensor = torch.randn(1, 4, 32, 64)
        await mgr.get_or_create("session_1", gpu_id=0, initial_kv=kv_tensor)
        
        freed_bytes = await mgr.close_session("session_1")
        
        assert freed_bytes > 0
        assert "session_1" not in mgr._sessions
        assert mgr.stats["sessions_closed"] == 1
    
    async def test_session_close_nonexistent(self):
        """Test closing nonexistent session returns 0."""
        mgr = KVSessionManager()
        freed_bytes = await mgr.close_session("nonexistent")
        assert freed_bytes == 0
    
    async def test_concurrent_sessions(self):
        """Test managing multiple concurrent sessions."""
        mgr = KVSessionManager()
        
        # Create multiple sessions
        for i in range(5):
            await mgr.get_or_create(f"session_{i}", gpu_id=i % 2)
        
        assert len(mgr._sessions) == 5
        assert mgr.stats["sessions_created"] == 5
        assert mgr.stats["max_concurrent_sessions"] == 5
    
    async def test_session_kv_retrieval(self):
        """Test getting KV without updating access time."""
        mgr = KVSessionManager()
        
        kv_tensor = torch.randn(1, 4, 32, 64)
        await mgr.get_or_create("session_1", gpu_id=0, initial_kv=kv_tensor)
        
        kv = await mgr.get_session_kv("session_1")
        assert kv is not None
        
        # Should not change step count
        assert mgr._sessions["session_1"].step_count == 1
    
    async def test_idle_cleanup(self):
        """Test idle session cleanup."""
        mgr = KVSessionManager(idle_timeout_seconds=0.1, cleanup_interval_seconds=0.05)
        
        # Create session
        await mgr.get_or_create("session_1", gpu_id=0)
        assert len(mgr._sessions) == 1
        
        # Start cleanup
        await mgr.start_cleanup()
        
        # Wait for cleanup
        await asyncio.sleep(0.3)
        
        # Session should be cleaned up
        assert len(mgr._sessions) == 0
        assert mgr.stats["cleanup_evictions"] == 1
        
        await mgr.stop_cleanup()
    
    async def test_cleanup_start_stop(self):
        """Test starting and stopping cleanup task."""
        mgr = KVSessionManager()
        
        assert mgr._cleanup_task is None
        
        await mgr.start_cleanup()
        assert mgr._cleanup_task is not None
        
        await mgr.stop_cleanup()
        assert mgr._cleanup_task is None
    
    async def test_manager_statistics(self):
        """Test manager statistics."""
        mgr = KVSessionManager()
        
        # Create session with KV
        kv_tensor = torch.randn(1, 4, 32, 64)
        await mgr.get_or_create("session_1", gpu_id=0, initial_kv=kv_tensor)
        
        stats = mgr.get_stats()
        
        assert stats["sessions_created"] == 1
        assert stats["active_sessions"] == 1
        assert stats["kv_bytes_pinned_mb"] > 0
    
    async def test_multiple_sessions_same_gpu(self):
        """Test multiple sessions pinned to same GPU."""
        mgr = KVSessionManager()
        
        # Create multiple sessions on GPU 0
        for i in range(3):
            await mgr.get_or_create(f"session_{i}", gpu_id=0)
        
        assert len(mgr._sessions) == 3
        
        # All should be on same GPU
        for i in range(3):
            assert mgr._sessions[f"session_{i}"].gpu_id == 0
    
    async def test_kv_bytes_tracking(self):
        """Test KV cache bytes tracking."""
        mgr = KVSessionManager()
        
        # Create session with KV
        kv_tensor1 = torch.randn(1, 4, 32, 64)
        await mgr.get_or_create("session_1", gpu_id=0, initial_kv=kv_tensor1)
        
        bytes_after_create = mgr.stats["kv_bytes_pinned"]
        
        # Update with larger KV
        kv_tensor2 = torch.randn(1, 8, 32, 64)
        await mgr.update_kv("session_1", kv_tensor2)
        
        bytes_after_update = mgr.stats["kv_bytes_pinned"]
        
        # Should increase
        assert bytes_after_update > bytes_after_create
    
    async def test_error_on_update_nonexistent(self):
        """Test error when updating nonexistent session."""
        mgr = KVSessionManager()
        
        kv_tensor = torch.randn(1, 4, 32, 64)
        
        with pytest.raises(ValueError, match="Session .* not found"):
            await mgr.update_kv("nonexistent", kv_tensor)


@pytest.mark.asyncio
class TestKVSessionConcurrency:
    """Test concurrent session operations."""
    
    async def test_concurrent_creation(self):
        """Test concurrent session creation."""
        mgr = KVSessionManager()
        
        # Create sessions concurrently
        tasks = [
            mgr.get_or_create(f"session_{i}", gpu_id=i % 2)
            for i in range(10)
        ]
        
        sessions = await asyncio.gather(*tasks)
        
        assert len(mgr._sessions) == 10
        assert len(sessions) == 10
    
    async def test_concurrent_updates(self):
        """Test concurrent KV updates."""
        mgr = KVSessionManager()
        
        # Create two sessions
        await mgr.get_or_create("session_1", gpu_id=0)
        await mgr.get_or_create("session_2", gpu_id=0)
        
        # Update concurrently
        kv1 = torch.randn(1, 4, 32, 64)
        kv2 = torch.randn(1, 4, 32, 64)
        
        tasks = [
            mgr.update_kv("session_1", kv1),
            mgr.update_kv("session_2", kv2),
        ]
        
        sessions = await asyncio.gather(*tasks)
        
        assert sessions[0].step_count > 1
        assert sessions[1].step_count > 1
    
    async def test_concurrent_cleanup_and_creation(self):
        """Test cleanup running while creating sessions."""
        mgr = KVSessionManager(idle_timeout_seconds=0.2, cleanup_interval_seconds=0.05)
        
        await mgr.start_cleanup()
        
        # Create sessions while cleanup is running
        for i in range(3):
            await mgr.get_or_create(f"session_{i}", gpu_id=0)
            await asyncio.sleep(0.08)
        
        await mgr.stop_cleanup()


@pytest.mark.asyncio
class TestKVSessionEdgeCases:
    """Test edge cases and error conditions."""
    
    async def test_session_without_kv(self):
        """Test session without KV cache."""
        mgr = KVSessionManager()
        
        session = await mgr.get_or_create("session_1", gpu_id=0, initial_kv=None)
        
        assert session.kv_cache is None
        assert session.bytes_used == 0
    
    async def test_empty_manager_stats(self):
        """Test stats when manager is empty."""
        mgr = KVSessionManager()
        
        stats = mgr.get_stats()
        
        assert stats["active_sessions"] == 0
        assert stats["sessions_created"] == 0
        assert stats["kv_bytes_pinned_mb"] == 0.0
    
    async def test_max_concurrent_sessions_tracking(self):
        """Test tracking of maximum concurrent sessions."""
        mgr = KVSessionManager()
        
        # Create 5 sessions
        for i in range(5):
            await mgr.get_or_create(f"session_{i}", gpu_id=0)
        
        assert mgr.stats["max_concurrent_sessions"] == 5
        
        # Close 2 sessions
        await mgr.close_session("session_0")
        await mgr.close_session("session_1")
        
        # Max should still be 5
        assert mgr.stats["max_concurrent_sessions"] == 5
        
        # But active should be 3
        assert len(mgr._sessions) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
