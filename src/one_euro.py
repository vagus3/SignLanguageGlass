# -*- coding: utf-8 -*-
"""
One Euro Filter

이동평균은 부드럽게 만들수록 지연이 그대로 늘어납니다.
One Euro 는 "천천히 움직일 땐 세게, 빠르게 움직일 땐 약하게" 필터링해서
같은 부드러움에서 지연이 훨씬 적습니다. AR 오버레이 흔들림 제거에 표준적으로 씁니다.

    f = OneEuroPoint(min_cutoff=1.0, beta=0.02)
    x, y = f(raw_x, raw_y, timestamp)
"""
import math


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """스칼라 1개용."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev = None
        self._t = None

    def __call__(self, x, t):
        if self._t is None:
            self._t, self._prev = t, x
            return self._x(x, 1.0)
        dt = max(t - self._t, 1e-3)
        self._t = t

        dx = (x - self._prev) / dt
        self._prev = x
        edx = self._dx(dx, _alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, _alpha(cutoff, dt))

    def reset(self):
        self.__init__(self.min_cutoff, self.beta, self.d_cutoff)


class OneEuroPoint:
    """(x, y) 좌표용."""

    def __init__(self, min_cutoff=1.0, beta=0.02):
        self.fx = OneEuro(min_cutoff, beta)
        self.fy = OneEuro(min_cutoff, beta)

    def __call__(self, x, y, t):
        return self.fx(x, t), self.fy(y, t)

    def reset(self):
        self.fx.reset()
        self.fy.reset()
