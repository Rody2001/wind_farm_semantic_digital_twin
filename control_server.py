"""Browser control panel for the wind farm simulation.

Serves a small HTML page (``control_panel.html``) that lets the user change the
environment parameters -- wind speed, wind direction, air temperature and the
grid limit -- while the MuJoCo simulation is running.

The existing keyboard controls keep working.  Both input paths write into the
same :class:`EnvironmentState` object, and the page polls the server, so a
change made with the keyboard shows up in the browser and the other way round.

Only the Python standard library is used, so the simulation keeps its previous
dependency set.

Standalone test (no MuJoCo needed)::

    python control_server.py

Integration::

    from control_server import EnvironmentState, start_control_server

    env = EnvironmentState(wind_speed=8.0, wind_direction=0.0,
                           temperature=15.0, grid_limit=500.0)
    start_control_server(env, port=8080)

    while running:
        e = env.snapshot()              # read once per step
        apply_wind(e["wind_speed"], e["wind_direction"], e["temperature"])
        ...
        env.set_telemetry(power=p, rpm=rpm)   # optional, shown in the panel
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

HTML_FILE = Path(__file__).with_name("control_panel.html")

# Single source of truth for the controls.  The browser builds the panel from
# this, so changing a range or a unit here is enough -- no edit in the HTML.
FIELDS: Dict[str, Dict[str, Any]] = {
    "wind_speed": {
        "label": "Wind speed",
        "unit": "m/s",
        "control": "slider",
        "min": 0.0, "max": 35.0, "step": 0.1, "decimals": 1,
        "hint": "Free-stream speed at hub height",
    },
    "wind_direction": {
        "label": "Wind direction",
        "unit": "\u00b0",
        "control": "compass",
        "min": 0.0, "max": 360.0, "step": 1.0, "decimals": 0,
        "wrap": True,
        "hint": "Meteorological: direction the wind comes from, 0\u00b0 = north",
    },
    "temperature": {
        "label": "Air temperature",
        "unit": "\u00b0C",
        "control": "slider",
        "min": -30.0, "max": 50.0, "step": 1.0, "decimals": 1,
        "hint": "Sets air density \u03c1 through the ideal gas law (15 \u00b0C \u2192 1.225 kg/m\u00b3)",
    },
    "grid_limit": {
        "label": "Grid limit",
        "unit": "MW",
        "control": "slider",
        "min": 0.0, "max": 1000.0, "step": 10.0, "decimals": 0,
        "hint": "Turbines are curtailed so the total stays under this value",
    },
}


def set_field_range(key: str, minimum: float = None, maximum: float = None,
                    step: float = None, decimals: int = None) -> Dict[str, Any]:
    """Change a control's range before the panel is served.

    EnvironmentState clamps to these numbers, so a value outside the range is not
    just un-draggable in the browser, it cannot be held at all. Widen the range
    first if the simulation needs a bigger one -- e.g. a grid limit above the
    default 1000 MW. The browser reads the range from /fields at page load, so
    nothing in the HTML needs changing.
    """
    spec = FIELDS[key]
    if minimum is not None:
        spec["min"] = float(minimum)
    if maximum is not None:
        spec["max"] = float(maximum)
    if step is not None:
        spec["step"] = float(step)
    if decimals is not None:
        spec["decimals"] = int(decimals)
    return spec


class EnvironmentState:
    """Thread-safe container for the environment parameters.

    The simulation thread reads with :meth:`snapshot`; the HTTP thread and the
    keyboard handler write with :meth:`update`.  ``version`` is incremented on
    every change so the browser can tell whether it needs to redraw.
    """

    def __init__(
        self,
        wind_speed: float = 8.0,
        wind_direction: float = 0.0,
        temperature: float = 15.0,
        grid_limit: float = 500.0,
    ) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, float] = {
            "wind_speed": float(wind_speed),
            "wind_direction": float(wind_direction),
            "temperature": float(temperature),
            "grid_limit": float(grid_limit),
        }
        for key in self._values:
            self._values[key] = _clamp(key, self._values[key])
        self._telemetry: Dict[str, Any] = {}
        self._version = 0
        self._source = "init"

    def snapshot(self) -> Dict[str, Any]:
        """Return a plain copy of the current values (safe to read per step)."""
        with self._lock:
            data = dict(self._values)
            data["version"] = self._version
            data["source"] = self._source
            data["telemetry"] = dict(self._telemetry)
            return data

    def update(self, source: str = "ui", **changes: float) -> Dict[str, Any]:
        """Set one or more values.  Unknown keys are ignored, values are clamped.

        ``source`` is only informational ("ui", "keyboard", ...) and is shown in
        the panel so it is obvious where the last change came from.
        """
        with self._lock:
            changed = False
            for key, raw in changes.items():
                if key not in self._values or raw is None:
                    continue
                try:
                    value = _clamp(key, float(raw))
                except (TypeError, ValueError):
                    continue
                if value != self._values[key]:
                    self._values[key] = value
                    changed = True
            if changed:
                self._version += 1
                self._source = source
        return self.snapshot()

    def set_telemetry(self, **values: Any) -> None:
        """Push read-only numbers (power, rpm, ...) to the panel.  Optional."""
        with self._lock:
            self._telemetry.update(values)

    # convenience accessors, so the sim code can stay readable
    @property
    def wind_speed(self) -> float:
        with self._lock:
            return self._values["wind_speed"]

    @property
    def wind_direction(self) -> float:
        with self._lock:
            return self._values["wind_direction"]

    @property
    def temperature(self) -> float:
        with self._lock:
            return self._values["temperature"]

    @property
    def grid_limit(self) -> float:
        with self._lock:
            return self._values["grid_limit"]


def _clamp(key: str, value: float) -> float:
    spec = FIELDS[key]
    if spec.get("wrap"):
        return value % spec["max"]
    return max(spec["min"], min(spec["max"], value))


class _Handler(BaseHTTPRequestHandler):
    """Three routes: the page, the field description, the values."""

    state: EnvironmentState  # injected by start_control_server
    quiet = True
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 (name fixed by BaseHTTPRequestHandler)
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                body = HTML_FILE.read_bytes()
            except OSError:
                self._send(503, b"control_panel.html not found next to control_server.py",
                           "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
        elif path == "/fields":
            self._send_json(200, FIELDS)
        elif path == "/state":
            self._send_json(200, self.state.snapshot())
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/state":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            self._send_json(400, {"error": "expected a JSON object of field values"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "expected a JSON object of field values"})
            return
        payload.pop("source", None)
        self._send_json(200, self.state.update(source="ui", **payload))

    def _send_json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away mid-response

    def log_message(self, fmt: str, *args: Any) -> None:
        if not self.quiet:
            super().log_message(fmt, *args)


def start_control_server(
    state: EnvironmentState,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = False,
    quiet: bool = True,
) -> ThreadingHTTPServer:
    """Start the panel in a daemon thread and return the running server.

    The thread is a daemon, so the simulation can exit without shutting the
    server down explicitly.  Call ``server.shutdown()`` for a clean stop.
    """
    handler = type("_BoundHandler", (_Handler,), {"state": state, "quiet": quiet})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="control-panel",
                     daemon=True).start()
    url = f"http://{host}:{port}/"
    print(f"[control panel] {url}")
    if open_browser:
        webbrowser.open(url)
    return server


if __name__ == "__main__":
    # Standalone check: serve the panel with a fake simulation behind it, so the
    # UI can be developed without starting MuJoCo.
    import math
    import time

    env = EnvironmentState()
    start_control_server(env, open_browser=False)
    print("Standalone demo running, Ctrl+C to stop.")
    try:
        while True:
            e = env.snapshot()
            # crude stand-in for the real power model, only to fill the readouts
            available = min(0.6 * e["wind_speed"] ** 3, 900.0)
            env.set_telemetry(
                power=round(min(available, e["grid_limit"]), 2),
                energy=round(time.monotonic() % 1000, 2),
                spinning="6/8",
                curtailed=available > e["grid_limit"],
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopped")