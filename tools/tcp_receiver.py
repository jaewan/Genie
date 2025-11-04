from __future__ import annotations

import asyncio
from typing import Tuple


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        # Read tensor_id length
        raw = await reader.readexactly(4)
        id_len = int.from_bytes(raw, "big")
        # Read tensor_id
        tensor_id = (await reader.readexactly(id_len)).decode("utf-8", errors="ignore")
        # Read payload length
        raw = await reader.readexactly(8)
        num_bytes = int.from_bytes(raw, "big")
        print(f"[recv] tensor_id={tensor_id} num_bytes={num_bytes}")
        remaining = num_bytes
        # Drain payload
        while remaining > 0:
            chunk = await reader.read(min(1 << 20, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        writer.close()
        try:
            await writer.wait_closed()
        except AttributeError:
            pass
        print(f"[done] tensor_id={tensor_id} consumed={num_bytes - remaining}")
    except asyncio.IncompleteReadError:
        print("[error] incomplete read; client disconnected early")


async def main(host: str = "0.0.0.0", port: int = 50555) -> None:
    server = await asyncio.start_server(handle_client, host, port)
    addr = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"[listening] {addr}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[shutdown]")


