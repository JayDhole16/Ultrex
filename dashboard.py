"""Read-only live dashboard for a training run's events.jsonl.

    python dashboard.py runs/<run_id>                  # then open http://127.0.0.1:8000
    python dashboard.py runs/<run_id>/events.jsonl --port 8080

Serves one page (dashboard.html) that polls this server for new lines of the run's events.jsonl, charts
population size and wealth inequality over generations, shows each living agent's token balance, and lists
the sampled negotiation transcripts. Its playback controls (play/pause, jump to a generation, follow live)
run entirely in the browser and only change what that page displays.

The server has no way to affect the run. It uses the standard library only and imports nothing from
train.py or the rest of this project. It opens one data file, events.jsonl, read-only, and never creates or
writes a file. It answers only GET and HEAD, on two routes:

    /                  the dashboard page
    /dashboard.js      its script
    /api/events        ?offset=N: the complete JSON lines after byte N of events.jsonl, and the next offset
    /api/live          the same, over live.jsonl: the move-by-move stream the arena animates

Other paths get 404, and any other HTTP method gets 501. It listens on 127.0.0.1 unless --host says otherwise.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PAGE = Path(__file__).with_name("dashboard.html")
SCRIPT = Path(__file__).with_name("dashboard.js")
MAX_CHUNK = 1 << 20  # bytes of log per response; the page asks again straight away while more remains
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    # The page may only run its own inline code and fetch from this server: no external requests, no forms.
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; connect-src 'self';"
        " base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}


def reject_constant(token):
    raise ValueError(f"{token} is not valid JSON")


def read_events(path, offset):
    """The parsed complete lines of the log after byte offset, and the offset to read from next.

    A last line without its newline is still being written, so it waits for the next read. If the file is
    now shorter than offset it has been replaced, and reading starts over from the top with reset set.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {"offset": 0, "events": [], "skipped": 0, "more": False, "reset": offset > 0, "waiting": True}
    reset = offset > size
    if reset:
        offset = 0
    with path.open("rb") as log:
        log.seek(offset)
        chunk = log.read(MAX_CHUNK)
        if b"\n" not in chunk:
            chunk += log.readline()  # a single line longer than a chunk: finish it
    complete = chunk[: chunk.rfind(b"\n") + 1]
    events, skipped = [], 0
    for line in complete.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line, parse_constant=reject_constant))
        except ValueError:
            skipped += 1
    next_offset = offset + len(complete)
    return {
        "offset": next_offset,
        "events": events,
        "skipped": skipped,
        "more": bool(complete) and next_offset < size,
        "reset": reset,
        "waiting": False,
    }


class Handler(BaseHTTPRequestHandler):
    """Answers GET and HEAD only; BaseHTTPRequestHandler replies 501 to every method without a do_ handler."""

    server_version = "RunDashboard/1"

    def do_GET(self):
        self.respond(send_body=True)

    def do_HEAD(self):
        self.respond(send_body=False)

    def respond(self, send_body):
        url = urlsplit(self.path)
        # The two files are re-read per request, so editing them only needs a browser refresh.
        if url.path == "/":
            self.send(HTTPStatus.OK, "text/html; charset=utf-8", PAGE.read_bytes(), send_body)
        elif url.path == "/dashboard.js":
            self.send(HTTPStatus.OK, "text/javascript; charset=utf-8", SCRIPT.read_bytes(), send_body)
        elif url.path in ("/api/events", "/api/live"):
            try:
                offset = int(parse_qs(url.query).get("offset", ["0"])[0])
                if offset < 0:
                    raise ValueError
            except ValueError:
                body = b"offset must be a non-negative integer"
                return self.send(HTTPStatus.BAD_REQUEST, "text/plain; charset=utf-8", body, send_body)
            path = self.server.events_path if url.path == "/api/events" else self.server.live_path
            payload = read_events(path, offset)
            payload["run"] = self.server.events_path.parent.name
            self.send(HTTPStatus.OK, "application/json", json.dumps(payload).encode(), send_body)
        else:
            self.send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found", send_body)

    def send(self, status, content_type, body, send_body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # the page polls every second; logging each request would bury everything else


def main():
    parser = argparse.ArgumentParser(description="Read-only live dashboard for a run's events.jsonl.")
    parser.add_argument("run", help="a run directory (runs/<run_id>) or its events.jsonl")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run = Path(args.run)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.events_path = (run if run.suffix == ".jsonl" else run / "events.jsonl").absolute()
    server.live_path = server.events_path.with_name("live.jsonl")  # the move-by-move stream, when the run writes one
    PAGE.read_bytes()  # fail here, not on the first request, if the page is missing
    SCRIPT.read_bytes()
    print(f"Dashboard for {server.events_path}")
    print(f"Open http://{args.host}:{args.port}/ (read-only; Ctrl+C stops the dashboard, not the run)")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Listening on {args.host}: anyone who can reach this port can read the run's log.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
