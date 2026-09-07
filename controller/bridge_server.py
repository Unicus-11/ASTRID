"""
bridge_server.py — live dual sumo-gui bridge (Normal vs Astrid RF, lockstep).
Run from ASTRID/controller/:  uvicorn bridge_server:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

THIS_DIR = Path(__file__).resolve().parent          # ASTRID/controller
REPO_ROOT = THIS_DIR.parent                          # ASTRID/
WORKER_SCRIPT = THIS_DIR / "sumo_worker.py"
SCENARIOS_ROOT = REPO_ROOT / "sumo" / "generated_scenarios"
SUMO_CONFIG_JSON = REPO_ROOT / "sumo" / "scenario_config.json"
FRONTEND_DIR = REPO_ROOT / "frontend" / "output"
SUMO_BINARY = "sumo-gui"

app = FastAPI()


def list_scenarios() -> list:
    if not SCENARIOS_ROOT.is_dir():
        return []
    return sorted(
        p.name for p in SCENARIOS_ROOT.iterdir()
        if p.is_dir() and (p / "scenario.json").is_file()
    )


class Worker:
    def __init__(self, role: str):
        self.role = role
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None

    async def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            sys.stderr.write(f"[{self.role}] {line.decode(errors='replace')}")

    async def start(self, scenario_name: str) -> dict:
        args = [
            sys.executable, str(WORKER_SCRIPT),
            "--role", self.role,
            "--scenario-dir", str(SCENARIOS_ROOT / scenario_name),
            "--sumo-config-json", str(SUMO_CONFIG_JSON),
            "--sumo-binary", SUMO_BINARY,
        ]
        self.proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(REPO_ROOT),
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        msg = await self._readline()
        if msg is None or msg.get("type") != "ready":
            err = msg.get("message") if msg else "worker died before sending 'ready'"
            raise RuntimeError(f"{self.role} worker failed to start: {err}")
        return msg

    async def _readline(self) -> Optional[dict]:
        assert self.proc is not None and self.proc.stdout is not None
        line = await self.proc.stdout.readline()
        if not line:
            return None
        try:
            return json.loads(line.decode("utf-8").strip())
        except json.JSONDecodeError:
            return None

    async def step(self) -> dict:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((json.dumps({"cmd": "step"}) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()
        msg = await self._readline()
        if msg is None:
            return {"type": "error", "message": f"{self.role} worker exited unexpectedly"}
        return msg

    async def stop(self) -> None:
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write((json.dumps({"cmd": "quit"}) + "\n").encode("utf-8"))
                await self.proc.stdin.drain()
            await asyncio.wait_for(self.proc.wait(), timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        self.proc = None


class Session:
    def __init__(self):
        self.normal = Worker("normal")
        self.astrid = Worker("astrid")
        self.scenario: Optional[str] = None
        self.sim_end_s: Optional[float] = None
        self.playing = False
        self.speed = 10
        self.play_task: Optional[asyncio.Task] = None
        self.requested_transitions = 0
        self.forced_transitions = 0

    async def load_scenario(self, name: str, ws: WebSocket) -> None:
        await self.pause()
        await self.normal.stop()
        await self.astrid.stop()
        self.requested_transitions = 0
        self.forced_transitions = 0

        try:
            normal_ready, _astrid_ready = await asyncio.gather(
                self.normal.start(name), self.astrid.start(name)
            )
        except Exception as exc:  # noqa: BLE001
            await ws.send_json({"type": "error", "message": str(exc)})
            return

        self.scenario = name
        self.sim_end_s = normal_ready.get("sim_end_s")
        await ws.send_json({
            "type": "scenario_loaded",
            "name": name,
            "sim_end_s": self.sim_end_s,
        })

    async def step_once(self, ws: WebSocket) -> bool:
        if self.normal.proc is None or self.astrid.proc is None:
            return False
        normal_msg, astrid_msg = await asyncio.gather(self.normal.step(), self.astrid.step())

        for msg in (normal_msg, astrid_msg):
            if msg.get("type") == "error":
                await ws.send_json(msg)
                await self.pause()
                return False

        if normal_msg.get("type") == "done" or astrid_msg.get("type") == "done":
            await ws.send_json({"type": "done"})
            return False

        normal_frame = normal_msg["frame"]
        astrid_frame = astrid_msg["frame"]
        if astrid_frame["resolved"] == "BEGIN_TRANSITION":
            self.requested_transitions += 1
        if astrid_frame["resolved"] == "FORCE_TRANSITION_MAX_GREEN":
            self.forced_transitions += 1

        await ws.send_json({
            "type": "frame",
            "normal": normal_frame,
            "astrid": astrid_frame,
            "requested_transitions": self.requested_transitions,
            "forced_transitions": self.forced_transitions,
        })
        return True

    async def play(self, ws: WebSocket) -> None:
        if self.playing:
            return
        self.playing = True

        async def loop():
            try:
                while self.playing:
                    interval_s = max(0.01, 0.2 / max(self.speed, 1))
                    ok = await self.step_once(ws)
                    if not ok:
                        self.playing = False
                        break
                    await asyncio.sleep(interval_s)
            except (WebSocketDisconnect, RuntimeError):
                self.playing = False

        self.play_task = asyncio.create_task(loop())

    async def pause(self) -> None:
        self.playing = False
        if self.play_task is not None:
            self.play_task.cancel()
            self.play_task = None

    async def shutdown(self) -> None:
        await self.pause()
        await self.normal.stop()
        await self.astrid.stop()


@app.get("/api/scenarios")
async def api_scenarios():
    return {"scenarios": list_scenarios()}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = cmd.get("cmd")
            if action == "load_scenario":
                await session.load_scenario(cmd["name"], ws)
            elif action == "play":
                await session.play(ws)
            elif action == "pause":
                await session.pause()
            elif action == "step":
                await session.pause()
                await session.step_once(ws)
            elif action == "set_speed":
                session.speed = max(1, min(100, int(cmd.get("value", 10))))
    except WebSocketDisconnect:
        pass
    finally:
        await session.shutdown()


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")