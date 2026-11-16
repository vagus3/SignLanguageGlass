"""First-person (glasses) view from 3D joints: geometry, units, rendering, composed examples."""
import _path  # noqa
import numpy as np
from src.ego_synth import EgoComposer, head_frame, project, render, to_mm

# A person facing the camera (-z), 2.3 m away: eyes at y=-300 mm, neck -150, hips +350.
T = 40
pose = np.full((T, 25, 3), np.nan)
pose[:, 15], pose[:, 16] = [-30, -300, 2300], [30, -300, 2300]       # right eye, left eye
pose[:, 2], pose[:, 5] = [-180, -150, 2300], [180, -150, 2300]       # right/left shoulder
pose[:, 1], pose[:, 8] = [0, -150, 2300], [0, 350, 2300]             # neck, mid hip
frame = head_frame(pose)
eye, right, up, forward = frame
assert np.allclose(forward, [0, 0, -1]) and np.allclose(up, [0, -1, 0]) and np.allclose(right, [-1, 0, 0])

# Units: some AI Hub clips are in metres; shoulder width tells them apart.
p_m, = to_mm(pose / 1000)
assert np.allclose(p_m[np.isfinite(p_m)], pose[np.isfinite(pose)])

# A hand in front of the chest (250 mm below the eyes, 300 mm forward) is seen low in the
# middle; one at the forehead is not seen by a camera tilted 35 degrees down.
chest = np.array([[0.0, -50, 2000]])
uv, ok = project(chest, frame, pitch=35, fov=80)
assert ok[0] and 0.4 < uv[0, 0] < 0.6 and uv[0, 1] > 0.5, uv
forehead = np.array([[0.0, -330, 2280]])
assert not project(forehead, frame, pitch=35, fov=80)[1][0]
# The wearer's right hand appears on the right of the image (no mirror).
assert project(np.array([[-150.0, -50, 2000]]), frame, pitch=35, fov=80)[0][0, 0] > 0.5
# Reaching forward brings a hand at chin height into view.
chin = np.array([[0.0, -230, 2250]])
assert not project(chin, frame, pitch=35, fov=80)[1][0]
assert project(chin, frame, pitch=35, fov=80, reach=250)[1][0]

# Rendering keeps hands only; a hand mostly outside the image counts as not detected.
hand = np.repeat(chest[None], T, 0) + np.linspace(-20, 20, 21)[None, :, None] * [1, 0, 0]
t = np.arange(T) / 30
seq = render([(t, hand, np.full((T, 21, 3), np.nan), frame)],
             {"pitch": 35, "fov": 80, "reach": 0})
assert seq.mask[:, :21].all() and not seq.mask[:, 21:].any()
assert seq.coordinates.shape == (T, 49, 2) and len(seq.timestamps) == T

# Composer: phrase text for listed glosses, spaced words otherwise, a no-sign target for rests.
row = {"id": "NIA_SL_WORD0001_REAL01_F", "signer": "REAL01", "glosses": ["아프다"],
       "intervals": [[0.4, 1.0]], "usable": [0], "frame": frame, "times": np.arange(60) / 30,
       "left": np.repeat(hand[:1], 60, 0), "right": np.full((60, 21, 3), np.nan)}
row2 = {**row, "id": "NIA_SL_WORD0002_REAL01_F", "glosses": ["약"]}
composer = EgoComposer([row, row2], [{"glosses": ["약", "아프다"], "text": "약 먹고 아파요."}])
rng = np.random.default_rng(0)
seen = {}
for _ in range(200):
    item = composer.sample(rng)
    seen[tuple(item["glosses"])] = item["text"]
    assert item["sequence"].mask[:, 21:].sum() == 0
assert seen[("약", "아프다")] == "약 먹고 아파요." and seen[("아프다",)] == "아프다."
assert seen[()] == EgoComposer.NO_SIGN
print("[OK] glasses view: head frame, units, projection, reach, rendering, composed targets")

# Seen from above the back of a hand, a finger curled under the palm is hidden.
from src.ego_synth import hidden_fingers
palm_down = np.zeros((1, 21, 3))
palm_down[0, :, 2] = 2000.0
palm_down[0, :, 1] = -50.0
palm_down[0, 5] = [-40, -50, 1960]; palm_down[0, 17] = [40, -50, 1960]; palm_down[0, 9] = [0, -50, 1950]
palm_down[0, 8] = [-40, 10, 1940]          # index tip curled below the palm (away from the eyes)
palm_down[0, 20] = [40, -90, 1940]         # little finger tip raised above it (towards the eyes)
hidden = hidden_fingers(palm_down, frame)
assert hidden[0, 8] and not hidden[0, 20] and not hidden[0, [0, 5, 9, 17]].any()
print("[OK] fingers hidden behind the palm")
