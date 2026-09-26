"""
Live verification script: Connects to ws://localhost:8000/ws/telemetry and verifies 10 packets.
"""

import asyncio
import json
import websockets
import sys

async def verify():
    uri = "ws://127.0.0.1:8000/ws/telemetry"
    print(f"Connecting to {uri}...")
    async with websockets.connect(uri) as ws:
        print("Connected! Reading 10 packets...")
        for i in range(10):
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"[{i+1}/10] t_ms={data['timestamp_ms']} RPM={data['variables'].get('engine_rpm')} CHT={data['variables'].get('cht')} Health={data['health']['overall_score']}%")
            assert "timestamp_ms" in data
            assert "variables" in data
            assert "health" in data
    print("Live WebSocket test succeeded!")

if __name__ == "__main__":
    asyncio.run(verify())
