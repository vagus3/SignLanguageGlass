"""안경 카메라(착용자 시점)에서 손만 찾아 49개 관절 형식으로 바꿉니다.

얼굴과 어깨가 보이지 않으므로 Holistic 대신 MediaPipe Hands 를 씁니다. Hands 의 왼손/오른손
판정은 셀카(거울 화면)를 가정하고, 손등이 보이는 착용자 시점에서는 흔들릴 수 있습니다. 그래서
판정은 손이 처음 나타날 때만 참고하고, 그 뒤로는 앞 장면에서 가장 가까운 손을 같은 손으로
이어 갑니다. 두 손이 서로 자리를 바꾸는 수어는 위치로 다시 맞춥니다(왼손이 화면 왼쪽).
"""
import numpy as np

from src.collect_glasses import from_hands
from src.pose_sequence import JOINTS

LEFT, RIGHT = slice(0, 21), slice(21, 42)


def _center(points, mask, part):
    m = mask[part]
    return points[part][m].mean(0) if m.sum() >= 5 else None


class HandTracker:
    """장면마다 판정이 바뀌어도 왼손·오른손이 뒤바뀌지 않게 이어 붙입니다."""

    def __init__(self, max_jump=0.25):
        self.max_jump, self.previous = max_jump, None   # 한 장면에 손이 움직일 수 있는 거리(화면 비율)

    def reset(self):
        self.previous = None

    def _cost(self, now, mapping):
        total = 0.0
        for a, b in mapping:
            if now[a] is not None:
                before = self.previous[b]
                total += (min(np.linalg.norm(now[a] - before), self.max_jump)
                          if before is not None else self.max_jump)
        return total

    def update(self, points, mask):
        points, mask = points.copy(), mask.copy()
        now = {"L": _center(points, mask, LEFT), "R": _center(points, mask, RIGHT)}
        if self.previous is not None:
            swap = self._cost(now, (("L", "R"), ("R", "L"))) < \
                self._cost(now, (("L", "L"), ("R", "R"))) - 1e-6
        else:       # 처음 나타날 때: 두 손이 판정과 반대쪽에 있으면 위치를 믿습니다
            swap = (now["L"] is not None and now["R"] is not None
                    and now["L"][0] > now["R"][0] + 0.15)
        if swap:
            points[LEFT], points[RIGHT] = points[RIGHT].copy(), points[LEFT].copy()
            mask[LEFT], mask[RIGHT] = mask[RIGHT].copy(), mask[LEFT].copy()
            now = {"L": now["R"], "R": now["L"]}
        self.previous = now if (now["L"] is not None or now["R"] is not None) else None
        return points, mask


class GlassesHands:
    """카메라 화면 -> (49개 관절 좌표, 보임 여부, Hands 결과). 몸 관절 7개는 항상 안 보임."""

    def __init__(self, mirrored=False):
        from src.landmarks import make_preview_hands
        self.hands, self.mirrored, self.tracker = make_preview_hands(), mirrored, HandTracker()

    def process(self, rgb):
        results = self.hands.process(rgb)
        points, mask = from_hands(results, image_is_mirrored=self.mirrored)
        if not mask.any():
            self.tracker.reset()
            return points, mask, results
        points, mask = self.tracker.update(points, mask)
        assert points.shape == (JOINTS, 2)
        return points, mask, results

    def close(self):
        self.hands.close()
