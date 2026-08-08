import asyncio
import importlib.util
from pathlib import Path
import unittest


SERVER_PATH = (
    Path(__file__).parents[1]
    / "inference_framework"
    / "realtime_serving"
    / "omni_realtime_server.py"
)


def load_server_module():
    spec = importlib.util.spec_from_file_location("omni_realtime_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AudioBufferTest(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_a_complete_chunk_without_padding(self):
        server = load_server_module()
        buffer = server.AudioBuffer()
        waiter = asyncio.create_task(buffer.wait_for_bytes(8))

        await buffer.push(b"\x01\x02\x03\x04")
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())

        await buffer.push(b"\x05\x06\x07\x08")
        await asyncio.wait_for(waiter, timeout=1)
        self.assertEqual(
            await buffer.pop_existing(8),
            b"\x01\x02\x03\x04\x05\x06\x07\x08",
        )


if __name__ == "__main__":
    unittest.main()
