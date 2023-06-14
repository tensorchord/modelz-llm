import asyncio
import multiprocessing as mp
import pathlib
import pickle
import socket
import struct
from typing import Any

from modelz_llm.log import logger


def run_server(path: str, barrier: mp.Barrier, cls, **kwargs):
    proc = mp.get_context("spawn").Process(
        target=Server, args=(path, barrier, cls), kwargs=kwargs, daemon=True
    )
    proc.start()
    return proc


class Server:
    def __init__(self, path: str, barrier: mp.Barrier, cls, **kwargs):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        try:
            self.func = cls(**kwargs)
        finally:
            barrier.wait()
        self.run()

    def run(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(str(self.path))
            sock.listen()
            while True:
                conn, addr = sock.accept()
                with conn:
                    logger.info("connected to a client: '%s'", addr)
                    while True:
                        num_bytes = conn.recv(4)
                        if not num_bytes:
                            logger.info("cannot receive data")
                            break
                        num = struct.unpack("!I", num_bytes)[0]
                        data = conn.recv(num)
                        try:
                            req = pickle.loads(data)
                            resp = self.func(req)
                        except Exception as err:
                            logger.warning(err)
                            resp = err
                        data = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
                        conn.sendall(struct.pack("!I", len(data)))
                        conn.sendall(data)

                    logger.info("closing the connection")


class Client:
    def __init__(self, path: str) -> None:
        self.path = path
        self.reader = self.writer = None

    async def request(self, req: Any) -> Any:
        if self.reader is None or self.writer is None:
            self.reader, self.writer = await asyncio.open_unix_connection(self.path)

        data = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        self.writer.write(struct.pack("!I", len(data)))
        self.writer.write(data)
        await self.writer.drain()

        num_bytes = await self.reader.read(4)
        num = struct.unpack("!I", num_bytes)[0]
        data = await self.reader.read(num)
        return pickle.loads(data)

    async def close(self):
        self.writer.close()
        await self.writer.wait_closed()
