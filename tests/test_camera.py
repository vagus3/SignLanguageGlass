"""카메라가 사라진 상태에서 read() 가 CPU 를 태우지 않는지 확인."""
import _path  # noqa: F401  (저장소 루트를 sys.path 에)
import sys, time
import src.camera as cam

calls = {"n": 0}
def fake_open(verbose=True):
    calls["n"] += 1
    return None                      # 항상 열기 실패
cam.open_camera = fake_open

r = cam.CameraReader.__new__(cam.CameraReader)
r.cap = None; r.max_retry = 5; r.reopen_sec = 1.0; r._fails = 0; r._next_open = 0.0

t0 = time.perf_counter()
n = 0
while time.perf_counter() - t0 < 1.0:
    ok, f = r.read(); assert not ok
    n += 1
el = time.perf_counter() - t0
print(f"  1초 동안 read() 호출 {n}회 (스핀이면 수십만 회)")
print(f"  open_camera 재시도 {calls['n']}회 (reopen_sec=1.0 이므로 1~2회여야 정상)")
assert n < 40, f"스핀 감지: {n}회"
assert calls["n"] <= 2, f"재연결 폭주: {calls['n']}회"
print("  [OK] 백오프 동작 확인")
