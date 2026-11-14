r"""hand_shape3d.py 정규화 검증 (기존 프로젝트 코드 수정 없음)

실행 (프로젝트 루트에서):
    python tools/validate_hand_shape3d.py
또는 이 파일을 C:\SignLanguageGlass 바로 아래에 저장했다면:
    python validate_hand_shape3d.py
"""
from pathlib import Path
import sys
import numpy as np

# 프로젝트 루트 또는 tools/ 폴더에서 모두 실행할 수 있게 경로 탐색
HERE = Path(__file__).resolve()
ROOT = next((p for p in (HERE.parent, *HERE.parents)
             if (p / 'src' / 'hand_shape3d.py').is_file()), None)
if ROOT is None:
    raise SystemExit('오류: src/hand_shape3d.py를 찾지 못했습니다. 이 파일을 프로젝트 루트나 tools/ 폴더에 넣으세요.')
sys.path.insert(0, str(ROOT))

from src.hand_shape3d import FEATURE_DIM, PAIR_I, PAIR_J, from_raw_hands, resample_descriptors  # noqa: E402

# 실제 손처럼 배치한 인공 3D 관절 좌표 21개 (단위: 미터로 가정)
LEFT = np.array([
    [0, 0, 0],
    [-.030, .020, 0], [-.050, .040, .005], [-.070, .060, .008], [-.085, .075, .010],
    [-.020, .040, 0], [-.023, .080, .005], [-.024, .110, .008], [-.025, .140, .010],
    [0, .045, 0], [0, .085, .002], [0, .120, .002], [0, .160, .003],
    [.020, .040, 0], [.025, .075, .003], [.030, .108, .005], [.035, .140, .008],
    [.040, .030, 0], [.050, .060, .005], [.056, .085, .008], [.060, .106, .010],
], dtype=np.float64)
RIGHT = LEFT.copy()
RIGHT[:, 0] = -RIGHT[:, 0]
RIGHT[:, 1] *= .92
RIGHT += np.array([.35, -.12, .03])
POINTS = np.stack([LEFT, RIGHT])


def pack(points, confidences=None):
    if confidences is None:
        confidences = np.ones((2, 21, 1), dtype=np.float64)
    return np.concatenate([points, confidences], axis=2)


def features(points, confidences=None):
    return from_raw_hands(pack(points, confidences))


def record_result(label, condition, difference=None):
    ok = bool(condition)
    line = f"[{'PASS' if ok else 'FAIL'}] {label}"
    if difference is not None:
        line += f' (최대 차이: {difference:.3e})'
    print(line)
    results.append(ok)
    lines.append(line)


results = []
lines = []
ref = features(POINTS)
record_result('양손 특징 차원 = 462', ref.shape == (462,) and FEATURE_DIM == 462)

# 1) 공간 위치만 이동: 거리 자체가 같으므로 동일해야 함
moved = features(POINTS + np.array([1.20, -.60, .45]))
d = float(np.max(np.abs(ref - moved)))
record_result('평행이동(+1.20, -0.60, +0.45) 후 특징 동일', np.allclose(ref, moved, rtol=1e-5, atol=1e-5), d)

# 2) Z축 기준 60도 회전: 강체 회전 후 거리 동일
angle = np.deg2rad(60.)
R = np.array([[np.cos(angle), -np.sin(angle), 0],
              [np.sin(angle),  np.cos(angle), 0],
              [0, 0, 1]], dtype=np.float64)
rotated = features(POINTS @ R.T)
d = float(np.max(np.abs(ref - rotated)))
record_result('3D 회전(Z축 60도) 후 특징 동일', np.allclose(ref, rotated, rtol=1e-5, atol=1e-5), d)

# 3) 모든 좌표를 2배 확대: 기준 길이도 2배라 비율 동일
scaled = features(POINTS * 2.)
d = float(np.max(np.abs(ref - scaled)))
record_result('손 크기 2배 확대 후 특징 동일', np.allclose(ref, scaled, rtol=1e-5, atol=1e-5), d)

# 4) 평행이동, 회전, 크기변화를 동시에 적용
combined = features((POINTS @ R.T) * 1.7 + [1.20, -.60, .45])
d = float(np.max(np.abs(ref - combined)))
record_result('이동+회전+크기변화를 동시에 적용해도 특징 동일', np.allclose(ref, combined, rtol=1e-5, atol=1e-5), d)

# 5) 0번 손목 ~ 9번 중지 MCP는 정규화 기준이므로 1.0
baseline_scale = float(np.linalg.norm(LEFT[9] - LEFT[0]))
record_result('손목~중지 MCP의 정규화 거리 = 1.0', np.isclose(ref[8], 1.0))
example1 = f'[예시] 기준 길이(손목~중지): 원본 {baseline_scale:.6f} -> 2배 확대 {2*baseline_scale:.6f}'
example2 = f'[예시] 관절 0~1 원본 거리: {np.linalg.norm(LEFT[0]-LEFT[1]):.6f} -> 2배 확대 {2*np.linalg.norm(LEFT[0]-LEFT[1]):.6f}'
example3 = f'[예시] 관절 0~1 정규화 거리: {ref[0]:.6f} -> 2배 확대 후 {scaled[0]:.6f}'
print(example1)
print(example2)
print(example3)
lines.extend([example1, example2, example3])

# 6) 관절 5번 결측: 해당 관절 마스크가 0, 관련 거리도 0
confidence = np.ones((2, 21, 1), np.float64)
confidence[0, 5, 0] = 0.
missing = features(POINTS, confidence)
pair_5 = (PAIR_I == 5) | (PAIR_J == 5)
record_result('5번 관절 결측 시 마스크(210+5) = 0', missing[210+5] == 0)
record_result('5번 관절이 포함된 거리 특징은 모두 0', np.all(missing[:210][pair_5] == 0))

# 7) 좌우 반사: 손 모양 거리만으로 반사 방향을 구별할 수 없다는 한계 확인
mirrored_points = POINTS.copy()
mirrored_points[:, :, 0] *= -1.0
mirrored = features(mirrored_points)
record_result('한 손의 거울 반사 방향은 구별되지 않음 (방식의 한계)',
              np.allclose(ref, mirrored, rtol=1e-5, atol=1e-5))

# 8) 프레임 길이 통일: 60프레임 -> 30프레임 x 462차원
seq = np.stack([features(POINTS + [0.001*i, 0., 0.]) for i in range(60)])
resampled = resample_descriptors(seq, length=30)
record_result('60프레임 -> (30,462) 리샘플링', resampled.shape == (30,462))

report = '\n'.join(lines + [f'\n총 {sum(results)}/{len(results)} 항목 통과'])
print('\n' + report.splitlines()[-1])
report_path = ROOT / 'normalization_validation_results.txt'
report_path.write_text(report + '\n', encoding='utf-8')
print(f'결과 저장: {report_path}')
if not all(results):
    sys.exit(1)
