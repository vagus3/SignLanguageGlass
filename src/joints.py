"""Joint definitions shared by the MediaPipe and AI Hub (OpenPose) input paths.

Hands: OpenPose and MediaPipe use the same 21 points in the same order
(0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 little).
Upper body: seven points both formats define with the same meaning; MediaPipe
has no neck, so it is derived from the shoulder midpoint. Faces are excluded
(70 vs 468 points, and an egocentric camera may see the interlocutor).
"""

N_HAND = 21

# Semantic order of the shared upper-body points.
BODY_NAMES = ["neck", "r_shoulder", "l_shoulder",
              "r_elbow", "l_elbow", "r_wrist", "l_wrist"]

# MediaPipe Pose (33) indices; neck is the shoulder midpoint.
MP_POSE = {"r_shoulder": 12, "l_shoulder": 11,
           "r_elbow": 14, "l_elbow": 13,
           "r_wrist": 16, "l_wrist": 15}

# OpenPose BODY_25 indices.
OP_POSE = {"neck": 1, "r_shoulder": 2, "l_shoulder": 5,
           "r_elbow": 3, "l_elbow": 6,
           "r_wrist": 4, "l_wrist": 7}


def person_from_frame(node):
    """Return the person dict of one AI Hub keypoint JSON frame (absorbs layout variants)."""
    if isinstance(node, dict):
        if "people" in node:
            people = node["people"]
            if isinstance(people, list) and people:
                return people[0]
            if isinstance(people, dict):
                return people
        if any(k.endswith("_keypoints_2d") for k in node):
            return node
    return None
