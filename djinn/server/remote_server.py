"""
Remote GPU Server for Distributed Execution

Simple network server that listens for tensor computation requests
and executes them on a local GPU, returning results over the network.

Used for benchmarking to verify that Djinn actually sends operations
across the network instead of executing locally.
"""

import asyncio
import json
import struct
import torch
import logging
from typing import Dict, Any
import io

logger = logging.getLogger(__name__)


class RemoteGPUServer:
    """
    Minimal remote GPU server for benchmarking.
    
    Listens on a TCP port and executes tensor operations received from clients.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.host = host
        self.port = port
        self.server = None
        self.bytes_received = 0
        self.bytes_sent = 0
        self.operations_count = 0
        
    async def start(self):
        """Start the server."""
        self.server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port
        )
        
        logger.info(f"🚀 Remote GPU Server listening on {self.host}:{self.port}")
        
        async with self.server:
            await self.server.serve_forever()
    
    async def handle_client(self, reader, writer):
        """Handle incoming client connection."""
        try:
            while True:
                # Read request size
                size_data = await reader.readexactly(4)
                request_size = struct.unpack('>I', size_data)[0]
                
                # Read request data
                request_data = await reader.readexactly(request_size)
                self.bytes_received += request_size + 4
                
                # Parse request
                request = json.loads(request_data.decode('utf-8'))
                
                # Execute operation
                try:
                    result = await self.execute_operation(request)
                    response = {
                        'status': 'success',
                        'result': result
                    }
                except Exception as e:
                    response = {
                        'status': 'error',
                        'error': str(e)
                    }
                
                # Send response
                response_json = json.dumps(response)
                response_data = response_json.encode('utf-8')
                
                # Write response size and data
                size_header = struct.pack('>I', len(response_data))
                writer.write(size_header + response_data)
                await writer.drain()
                
                self.bytes_sent += len(response_data) + 4
                self.operations_count += 1
                
        except asyncio.IncompleteReadError:
            pass  # Client disconnected
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def execute_operation(self, request: Dict[str, Any]) -> str:
        """Execute a tensor operation and return result as base64."""
        operation = request.get('operation')
        operands = request.get('operands', [])
        kwargs = request.get('kwargs', {})
        
        # Deserialize operands
        tensors = []
        for operand_b64 in operands:
            # Simple base64 deserialization (in real code, would use pickle)
            import base64
            operand_data = base64.b64decode(operand_b64)
            buffer = io.BytesIO(operand_data)
            tensor = torch.load(buffer)
            tensors.append(tensor)
        
        # Execute operation
        if operation == 'matmul':
            result = tensors[0] @ tensors[1]
        elif operation == 'add':
            result = tensors[0] + tensors[1]
        elif operation == 'argmax':
            result = torch.argmax(tensors[0], **kwargs)
        elif operation == 'sum':
            result = torch.sum(tensors[0], **kwargs)
        elif operation == 'mean':
            result = torch.mean(tensors[0], **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Serialize result
        buffer = io.BytesIO()
        torch.save(result, buffer)
        import base64
        result_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return result_b64
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'bytes_received': self.bytes_received,
            'bytes_sent': self.bytes_sent,
            'operations': self.operations_count,
        }


def main():
    """Run the remote GPU server."""
    logging.basicConfig(level=logging.INFO)
    
    server = RemoteGPUServer(host="127.0.0.1", port=5555)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")


if __name__ == "__main__":
    main()

