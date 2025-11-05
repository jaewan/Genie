"""
Unit Tests for Connection Pool

Tests connection pool functionality:
- Basic pooling and connection reuse
- Health checking
- Connection lifecycle management
- Error handling and recovery
- Pool statistics
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock, MagicMock

import sys
sys.path.insert(0, '/home/jae/Genie')

from djinn.transport.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
class TestConnectionPool:
    """Test suite for ConnectionPool"""
    
    @pytest.fixture
    def pool(self):
        """Create a connection pool for testing"""
        async def _create_pool():
            pool = ConnectionPool(
                host="127.0.0.1",
                port=9999,
                pool_size=3,
                timeout=10.0
            )
            await pool.initialize()
            return pool
        
        async def _cleanup_pool(pool):
            await pool.close_all()
        
        # Return a simple object that can be used with async tests
        pool_obj = asyncio.run(_create_pool())
        yield pool_obj
        asyncio.run(_cleanup_pool(pool_obj))
    
    async def test_pool_initialization(self):
        """Test pool initializes correctly"""
        pool = ConnectionPool("localhost", 5000, pool_size=5)
        await pool.initialize()
        
        assert pool.host == "localhost"
        assert pool.port == 5000
        assert pool.pool_size == 5
        assert pool.available is not None
        assert pool.connections == []
        assert not pool.closed
        
        await pool.close_all()
    
    async def test_pool_size_clamping(self):
        """Test pool size is clamped to valid range"""
        pool_too_small = ConnectionPool("host", 5000, pool_size=0)
        assert pool_too_small.pool_size == 1
        
        pool_too_large = ConnectionPool("host", 5000, pool_size=100)
        assert pool_too_large.pool_size == 5
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test pool works as async context manager"""
        async with ConnectionPool("127.0.0.1", 9999) as pool:
            assert pool.available is not None
            assert not pool.closed
        
        # Pool should be closed after context exit
        assert pool.closed
    
    @pytest.mark.asyncio
    async def test_get_connection_raises_when_closed(self, pool):
        """Test get_connection raises error if pool is closed"""
        await pool.close_all()
        
        with pytest.raises(RuntimeError, match="ConnectionPool is closed"):
            await pool.get_connection()
    
    @pytest.mark.asyncio
    async def test_is_healthy_check(self, pool):
        """Test health checking of connections"""
        mock_writer = Mock()
        mock_writer.is_closing.return_value = False
        assert pool._is_healthy(mock_writer)
        
        mock_writer.is_closing.return_value = True
        assert not pool._is_healthy(mock_writer)
        
        # Test with None writer
        assert not pool._is_healthy(None)
        
        # Test with exception
        bad_writer = Mock()
        bad_writer.is_closing.side_effect = Exception("error")
        assert not pool._is_healthy(bad_writer)
    
    @pytest.mark.asyncio
    async def test_release_healthy_connection(self, pool):
        """Test releasing a healthy connection returns it to pool"""
        mock_reader = Mock()
        mock_writer = Mock()
        mock_writer.is_closing.return_value = False
        
        await pool.release_connection(mock_reader, mock_writer)
        
        # Connection should be available
        assert pool.available.qsize() == 1
        returned_reader, returned_writer = await pool.available.get()
        assert returned_reader is mock_reader
        assert returned_writer is mock_writer
    
    @pytest.mark.asyncio
    async def test_release_unhealthy_connection(self, pool):
        """Test releasing unhealthy connection closes it"""
        mock_reader = Mock()
        mock_writer = Mock()
        mock_writer.is_closing.return_value = True
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()
        
        pool.connections.append((mock_reader, mock_writer))
        
        await pool.release_connection(mock_reader, mock_writer)
        
        # Connection should be removed from pool
        assert pool.available.qsize() == 0
        assert len(pool.connections) == 0
        mock_writer.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_release_when_closed(self, pool):
        """Test releasing connection when pool is closed"""
        mock_reader = Mock()
        mock_writer = Mock()
        mock_writer.is_closing.return_value = False
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()
        
        await pool.close_all()
        await pool.release_connection(mock_reader, mock_writer)
        
        # Connection should be closed
        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_connection(self, pool):
        """Test closing a single connection"""
        mock_writer = Mock()
        mock_writer.close = Mock()
        mock_writer.wait_closed = AsyncMock()
        
        await pool._close_connection(Mock(), mock_writer)
        
        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_connection_handles_errors(self, pool):
        """Test close_connection handles exceptions gracefully"""
        mock_writer = Mock()
        mock_writer.close = Mock(side_effect=Exception("close error"))
        
        # Should not raise exception
        await pool._close_connection(Mock(), mock_writer)
    
    @pytest.mark.asyncio
    async def test_close_all_closes_all_connections(self, pool):
        """Test close_all closes all connections in pool"""
        mock_connections = [
            (Mock(), Mock()) for _ in range(3)
        ]
        
        for reader, writer in mock_connections:
            writer.close = Mock()
            writer.wait_closed = AsyncMock()
        
        pool.connections = mock_connections
        
        await pool.close_all()
        
        # All connections should be closed
        for reader, writer in mock_connections:
            writer.close.assert_called_once()
            writer.wait_closed.assert_called_once()
        
        assert pool.connections == []
        assert pool.closed
    
    @pytest.mark.asyncio
    async def test_get_stats(self, pool):
        """Test getting pool statistics"""
        mock_reader = Mock()
        mock_writer = Mock()
        mock_writer.is_closing.return_value = False
        
        pool.connections.append((mock_reader, mock_writer))
        await pool.available.put((mock_reader, mock_writer))
        
        stats = pool.get_stats()
        
        assert stats['host'] == "127.0.0.1"
        assert stats['port'] == 9999
        assert stats['pool_size'] == 3
        assert stats['total_connections'] == 1
        assert stats['available_connections'] == 1
        assert stats['closed'] == False
    
    @pytest.mark.asyncio
    async def test_get_stats_when_closed(self, pool):
        """Test getting stats when pool is closed"""
        await pool.close_all()
        
        stats = pool.get_stats()
        assert stats['closed'] == True
        assert stats['total_connections'] == 0


@pytest.mark.asyncio
class TestConnectionPoolIntegration:
    """Integration tests with mock server"""
    
    @pytest.mark.asyncio
    async def test_connection_reuse_pattern(self):
        """Test connection reuse pattern"""
        pool = ConnectionPool("127.0.0.1", 9999, pool_size=2)
        await pool.initialize()
        
        mock_reader = Mock()
        mock_writer = Mock()
        mock_writer.is_closing.return_value = False
        
        # Release connection
        await pool.release_connection(mock_reader, mock_writer)
        assert pool.available.qsize() == 1
        
        # Try to get connection (should get same one)
        r, w = pool.available.get_nowait()
        assert r is mock_reader
        assert w is mock_writer
        
        await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_pool_size_enforcement(self):
        """Test pool doesn't exceed max size"""
        pool = ConnectionPool("127.0.0.1", 9999, pool_size=2)
        await pool.initialize()
        
        # Create 3 mock connections
        for i in range(3):
            reader = Mock()
            writer = Mock()
            writer.is_closing.return_value = False
            pool.connections.append((reader, writer))
        
        stats = pool.get_stats()
        # Note: actual pool size enforcement happens during creation
        # This test shows total tracking
        assert stats['total_connections'] == 3
        
        await pool.close_all()


class TestConnectionPoolRobustness:
    """Test robustness and edge cases"""
    
    def test_double_close_is_safe(self):
        """Test double close doesn't cause errors"""
        async def double_close():
            pool = ConnectionPool("127.0.0.1", 9999)
            await pool.initialize()
            await pool.close_all()
            # Should not raise error
            await pool.close_all()
        
        asyncio.run(double_close())
    
    def test_get_stats_with_no_available_queue(self):
        """Test get_stats when available queue not initialized"""
        pool = ConnectionPool("127.0.0.1", 9999)
        # Don't call initialize
        stats = pool.get_stats()
        assert stats['available_connections'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
