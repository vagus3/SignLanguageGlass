# -*- coding: utf-8 -*-
"""같은 네트워크에서 말풍선 영상을 읽기 전용으로 봅니다."""
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2  # 기존 영상 처리에 사용하는 OpenCV


def _local_ip():
    """기본 네트워크의 주소를 알아냅니다. 실제 데이터를 보내지는 않습니다."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class WebView:
    def __init__(self, port=8765, host="0.0.0.0"):
        self.token = secrets.token_urlsafe(24)
        self._condition = threading.Condition()
        self._frame = None
        self._jpeg = None
        self._revision = 0
        self._viewers = 0
        self._connections = set()
        self._closed = False
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # 접속 주소의 토큰을 로그에 남기지 않습니다.

            def do_GET(self):
                if self.path in (f"/{view.token}", f"/{view.token}/"):
                    page = ("<!doctype html><html lang='ko'><head>"
                            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                            "<meta name='referrer' content='no-referrer'>"
                            "<title>말풍선 화면</title><style>"
                            "html,body{margin:0;width:100%;height:100%;background:#000}"
                            "body{display:flex;align-items:center;justify-content:center}"
                            "img{width:100%;height:100%;object-fit:contain}"
                            "</style></head><body><img alt='말풍선 영상' src='/"
                            + view.token + "/stream'></body></html>").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(page)
                elif self.path == f"/{view.token}/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self):
                with view._condition:
                    if view._closed:
                        return
                    view._viewers += 1
                try:
                    self.connection.settimeout(2)
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    revision = 0
                    while True:
                        with view._condition:
                            view._condition.wait_for(
                                lambda: view._closed or view._revision != revision)
                            if view._closed:
                                break
                            revision, jpeg = view._revision, view._jpeg
                        if jpeg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                             + jpeg + b"\r\n")
                            self.wfile.flush()
                except (OSError, ConnectionError):
                    pass
                finally:
                    with view._condition:
                        view._viewers -= 1
                        if not view._viewers:
                            view._frame = None
                            view._jpeg = None

        class Server(ThreadingHTTPServer):
            daemon_threads = False
            allow_reuse_address = True

            def get_request(self):
                conn, address = super().get_request()
                conn.settimeout(2)
                with view._condition:
                    view._connections.add(conn)
                return conn, address

            def process_request_thread(self, request, client_address):
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    with view._condition:
                        view._connections.discard(request)

        self._server = Server((host, port), Handler)
        self.port = self._server.server_address[1]
        self._server_thread = threading.Thread(target=self._server.serve_forever,
                                               name="web-view-server")
        self._encoder_thread = threading.Thread(target=self._encode, name="web-view-encoder")

    def start(self):
        try:
            self._encoder_thread.start()
            self._server_thread.start()
        except Exception:
            self.close()
            raise
        print(f"[화면 공유] 휴대폰에서 열 주소: http://{_local_ip()}:{self.port}/{self.token}",
              flush=True)

    def update(self, frame):
        """보는 사람이 있을 때만 프레임을 복사하고 인코딩을 맡깁니다."""
        with self._condition:
            if self._viewers and not self._closed:
                self._frame = frame.copy()
                self._condition.notify_all()

    def _encode(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._frame is not None)
                if self._closed:
                    return
                frame, self._frame = self._frame, None
            ok, encoded = cv2.imencode(".jpg", frame)
            if ok:
                with self._condition:
                    if self._viewers and not self._closed:
                        self._jpeg = encoded.tobytes()
                        self._revision += 1
                        self._condition.notify_all()

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
            connections = list(self._connections)
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._server_thread.is_alive():
            self._server.shutdown()
        self._server.server_close()
        if self._server_thread.ident is not None:
            self._server_thread.join()
        if self._encoder_thread.ident is not None:
            self._encoder_thread.join()
