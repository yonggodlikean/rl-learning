"""Local question-bank HTTP server for the RL checkpoint (EasyRL v1.0.6 2.1-2.2.2).

Security posture (must hold):
  * Binds 127.0.0.1 only, on a fixed document root (this directory).
  * Serves only an allow-list of static files and local vendor assets.
  * Never executes user-submitted Python on the server (code grading is
    browser-side in a Pyodide worker).
  * Never exposes PDFs, parent paths, or arbitrary filesystem locations.

Run:  python3 server.py [--port 8000] [--root <dir>]
Endpoints:
  GET  /api/health
  GET  /api/questions
  POST /api/grade    {"id": ..., "answer": ...}
  POST /api/reveal   {"id": ...}
  GET  /api/config   (relative vendor asset availability)
"""

import argparse
import json
import os
import posixpath
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import bank
import chapters

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY_BYTES = 64 * 1024

PUBLIC_FILES = (
    "index.html",
    "app.js",
    "worker.js",
    "styles.css",
)

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".map": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".zip": "application/zip",
    ".whl": "application/octet-stream",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}

VENDOR_ALLOWED_ROOTS = ("vendor/pyodide", "vendor/katex", "vendor/editor")


class BadRequest(Exception):
    pass


def safe_static_path(root, url_path):
    """Resolve a static URL path to an absolute file path, or return None.

    Accepts only the allow-listed top-level files and files under
    vendor/pyodide/, vendor/katex/ or vendor/editor/. Rejects traversal
    and all other paths.
    """
    if not url_path.startswith("/"):
        return None
    rel = posixpath.normpath(url_path.lstrip("/"))
    if rel in (".", ""):
        rel = "index.html"
    parts = rel.split("/")
    if ".." in parts or rel.startswith("/") or os.path.isabs(rel):
        return None
    if rel in PUBLIC_FILES:
        pass
    elif any(rel.startswith(prefix + "/") for prefix in VENDOR_ALLOWED_ROOTS):
        if ".." in parts:
            return None
    else:
        return None
    candidate = os.path.normpath(os.path.join(root, rel))
    root_abs = os.path.abspath(root)
    candidate_abs = os.path.abspath(candidate)
    if candidate_abs != root_abs and not candidate_abs.startswith(root_abs + os.sep):
        return None
    return candidate_abs


def parse_grade_payload(payload):
    if not isinstance(payload, dict):
        raise BadRequest("请求体必须是 JSON 对象")
    question_id = payload.get("id")
    if not isinstance(question_id, str) or not question_id:
        raise BadRequest("缺少字段 id")
    if "answer" not in payload:
        raise BadRequest("缺少字段 answer")
    return question_id, payload["answer"]


def parse_reveal_payload(payload):
    if not isinstance(payload, dict):
        raise BadRequest("请求体必须是 JSON 对象")
    question_id = payload.get("id")
    if not isinstance(question_id, str) or not question_id:
        raise BadRequest("缺少字段 id")
    return question_id


def parse_chapter_payload(payload):
    chapter_id = payload.get("chapter")
    if chapter_id is not None and (not isinstance(chapter_id, str) or not chapter_id):
        raise BadRequest("chapter 必须是非空字符串")
    return chapter_id


def make_handler(root):
    root = os.path.abspath(root)

    class Handler(BaseHTTPRequestHandler):
        server_version = "RLCheckpoint/1.0"
        protocol_version = "HTTP/1.1"

        # ---- helpers -------------------------------------------------
        def _send_json(self, status, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_error_json(self, status, message):
            self._send_json(status, {"error": message})

        def _read_json_body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except (TypeError, ValueError):
                raise BadRequest("Content-Length 无效")
            if length <= 0:
                raise BadRequest("请求体为空")
            if length > MAX_BODY_BYTES:
                raise BadRequest("请求体过大")
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise BadRequest("请求体不是合法 JSON")

        def _path_only(self):
            return self.path.split("?", 1)[0].split("#", 1)[0]

        def _query_chapter(self):
            selected = parse_qs(urlsplit(self.path).query, keep_blank_values=True).get("chapter")
            if selected is None:
                return None
            if len(selected) != 1 or not selected[0]:
                raise BadRequest("chapter 查询参数必须恰好是一个非空字符串")
            return selected[0]

        # ---- routing -------------------------------------------------
        def do_GET(self):
            path = self._path_only()
            if path == "/api/health":
                self._send_json(200, bank.health())
                return
            if path == "/api/chapters":
                self._send_json(200, {
                    "default_chapter_id": chapters.DEFAULT_CHAPTER_ID,
                    "chapters": chapters.list_chapters(),
                })
                return
            if path == "/api/questions":
                try:
                    chapter_id = self._query_chapter()
                    selected = chapters.resolve_bank(chapter_id)
                except BadRequest as exc:
                    self._send_error_json(400, str(exc))
                    return
                except chapters.UnknownChapter as exc:
                    self._send_error_json(404, str(exc))
                    return
                self._send_json(200, selected.public_payload())
                return
            if path == "/api/config":
                self._send_json(200, self._config())
                return
            if path.startswith("/api/"):
                self._send_error_json(404, "未知接口")
                return
            self._serve_static(path)

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            path = self._path_only()
            if path == "/api/grade":
                try:
                    payload = self._read_json_body()
                    qid, answer = parse_grade_payload(payload)
                    chapter_id = parse_chapter_payload(payload)
                except BadRequest as exc:
                    self._send_error_json(400, str(exc))
                    return
                try:
                    selected = chapters.resolve_bank(chapter_id)
                    result = selected.grade(qid, answer)
                except chapters.UnknownChapter as exc:
                    self._send_error_json(404, str(exc))
                    return
                except bank.UnknownQuestion as exc:
                    self._send_error_json(404, str(exc))
                    return
                except bank.NotAutoGradable as exc:
                    self._send_error_json(422, str(exc))
                    return
                except bank.InvalidAnswer as exc:
                    self._send_error_json(400, str(exc))
                    return
                self._send_json(200, result)
                return
            if path == "/api/reveal":
                try:
                    payload = self._read_json_body()
                    qid = parse_reveal_payload(payload)
                    chapter_id = parse_chapter_payload(payload)
                except BadRequest as exc:
                    self._send_error_json(400, str(exc))
                    return
                try:
                    selected = chapters.resolve_bank(chapter_id)
                    result = selected.reveal(qid)
                except chapters.UnknownChapter as exc:
                    self._send_error_json(404, str(exc))
                    return
                except bank.UnknownQuestion as exc:
                    self._send_error_json(404, str(exc))
                    return
                except bank.NotRevealable as exc:
                    self._send_error_json(422, str(exc))
                    return
                self._send_json(200, result)
                return
            self._send_error_json(404, "未知接口")

        # ---- static --------------------------------------------------
        def _config(self):
            vendor_dir = os.path.join(root, "vendor", "pyodide")
            files = []
            if os.path.isdir(vendor_dir):
                for name in ("pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm",
                             "pyodide-lock.json", "python_stdlib.zip"):
                    if os.path.isfile(os.path.join(vendor_dir, name)):
                        files.append("%s/%s" % (VENDOR_ALLOWED_ROOTS[0], name))
            return {
                "vendor_pyodide": VENDOR_ALLOWED_ROOTS[0] if files else None,
                "vendor_available": bool(files),
                "files": files,
            }

        def _serve_static(self, path):
            target = safe_static_path(root, path)
            if target is None:
                self._send_error_json(404, "未找到资源")
                return
            if not os.path.isfile(target):
                self._send_error_json(404, "未找到资源")
                return
            try:
                with open(target, "rb") as fh:
                    body = fh.read()
            except OSError:
                self._send_error_json(404, "未找到资源")
                return
            ext = os.path.splitext(target)[1].lower()
            ctype = MIME_TYPES.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def log_message(self, fmt, *args):  # quieter default logging
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def build_server(host=DEFAULT_HOST, port=DEFAULT_PORT, root=None):
    if root is None:
        root = os.path.dirname(os.path.abspath(__file__))
    handler = make_handler(root)
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RL checkpoint local server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("只允许绑定 127.0.0.1")

    this_dir = os.path.dirname(os.path.abspath(__file__))
    httpd = build_server(args.host, args.port, args.root)
    print("RL checkpoint server on http://%s:%d/ (root=%s)"
          % (args.host, httpd.server_address[1], os.path.abspath(args.root or this_dir)))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
