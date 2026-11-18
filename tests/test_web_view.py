# -*- coding: utf-8 -*-
"""카메라 없이 휴대폰 영상 공유와 종료를 확인합니다."""
import _path  # noqa: F401
import http.client
import socket
import time
from unittest.mock import patch

import cv2
import numpy as np

from src.web_view import WebView


def connect(port):
    return http.client.HTTPConnection("127.0.0.1", port, timeout=3)


def main():
    view = WebView(0, host="127.0.0.1")
    view.start()
    stream = None
    idle = None
    try:
        frame = np.full((32, 48, 3), (20, 80, 180), dtype=np.uint8)
        with patch("src.web_view.cv2.imencode", wraps=cv2.imencode) as encode:
            view.update(frame)
            time.sleep(0.05)
            assert encode.call_count == 0

            conn = connect(view.port)
            conn.request("GET", "/wrong")
            response = conn.getresponse()
            assert response.status == 404
            response.read()
            conn.close()

            conn = connect(view.port)
            conn.request("GET", "/" + view.token)
            response = conn.getresponse()
            assert response.status == 200
            assert b"/stream" in response.read()
            conn.close()

            stream = connect(view.port)
            stream.request("GET", "/" + view.token + "/stream")
            response = stream.getresponse()
            assert response.status == 200
            assert response.getheader("Content-Type") == "multipart/x-mixed-replace; boundary=frame"
            view.update(frame)
            assert response.readline() == b"--frame\r\n"
            assert response.readline() == b"Content-Type: image/jpeg\r\n"
            length = int(response.readline().split(b":", 1)[1])
            assert response.readline() == b"\r\n"
            jpeg = response.read(length)
            assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert decoded.shape == frame.shape
            assert encode.call_count >= 1

        # 요청을 끝내지 않은 연결도 서버 종료를 막지 않아야 합니다.
        idle = socket.create_connection(("127.0.0.1", view.port), timeout=3)
    finally:
        view.close()
        if stream is not None:
            stream.close()
        if idle is not None:
            idle.close()
    assert not view._server_thread.is_alive()
    assert not view._encoder_thread.is_alive()
    with socket.socket() as sock:
        sock.settimeout(1)
        assert sock.connect_ex(("127.0.0.1", view.port)) != 0
    view.close()
    print("웹 화면 공유 테스트 통과")


if __name__ == "__main__":
    main()
