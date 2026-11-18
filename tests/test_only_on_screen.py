"""--only-on-screen: sound with no moving mouth is not transcribed; mid-turn gaps continue."""
import _path  # noqa: F401
import unittest

from src.speaker_audio import SpeakerAudioRouter


class Decoder:
    def __init__(self):
        self.fed = 0

    def feed(self, pcm):
        self.fed += 1
        return []

    def flush(self):
        return []

    def reset(self):
        pass


class Speakers:
    def __init__(self, owners):
        self.owners = list(owners)

    def resolve(self, at):
        return self.owners.pop(0)


def slices(n):
    return [(bytes(640), i * .02, (i + 1) * .02, True) for i in range(n)]


class OnScreenTests(unittest.TestCase):
    def run_router(self, owners, only):
        decoder, events = Decoder(), []
        router = SpeakerAudioRouter(decoder, lambda *a, **k: events.append(a), Speakers(owners),
                                    vad=object(), only_on_screen=only)
        router.feed(slices(len(owners)))
        return decoder.fed

    def test_unowned_sound_ignored_only_when_enabled(self):
        self.assertEqual(self.run_router([None] * 5, True), 0)
        self.assertEqual(self.run_router([None] * 5, False), 5)

    def test_gap_inside_turn_continues(self):
        self.assertEqual(self.run_router([1, None, None, 1], True), 4)

    def test_owner_change_still_splits(self):
        self.assertEqual(self.run_router([1, 2], True), 2)


if __name__ == "__main__":
    unittest.main()
