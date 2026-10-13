"""Two clocks with separate jobs; never subtract one from the other.

now(): elapsed time inside this process. Capture timestamps, segment timing,
cooldowns, latency spans, TTS start and the audio gate all use it, so a
system clock correction cannot reorder frames, keep the microphone muted or
produce negative latency. perf_counter rather than time.monotonic because on
Windows before Python 3.13 monotonic ticks every 15.6 ms (gh-88494), which
would give two 30 fps frames the same timestamp.

unix_time(): calendar time for data and logs only (e.g. capture_start).
"""
import time


def now():
    return time.perf_counter()


def unix_time():
    return time.time()
