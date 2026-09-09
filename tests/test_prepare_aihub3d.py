"""다중 시점 중복·누락 프레임이 평가 표본을 부풀리지 않는지 검증."""
import _path  # noqa: F401
import json
from pathlib import Path
import tempfile
import numpy as np
from tools.prepare_aihub3d import prepare

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / 'source'
    rng = np.random.default_rng(9)
    frames = [np.c_[rng.normal(size=(21,3)), np.ones(21)].ravel().tolist()
              for _ in range(3)]
    for view, indices in [('F',[0,1,2]), ('L',[0,2])]:
        folder = root / f'NIA_SL_WORD1501_REAL01_{view}'
        folder.mkdir(parents=True)
        for i in indices:
            node = {'people': {'hand_left_keypoints_3d': frames[i],
                               'hand_right_keypoints_3d': frames[i]}}
            (folder / f'{folder.name}_{i:012d}_keypoints.json').write_text(json.dumps(node))
    out = Path(td) / 'processed'
    report = prepare(root, out)
    assert report['unique_motion_groups'] == 1
    assert report['unique_3d_sequences'] == 1
    assert report['duplicate_3d_clips_removed'] == 1
    assert report['incomplete_clips'][0]['missing_frame_indices'] == [1]
    assert not report['can_make_disjoint_train_test_with_all_classes']
    with np.load(out / 'dataset.npz', allow_pickle=False) as d:
        assert d['X'].shape == (1,30,462) and np.isfinite(d['X']).all()
    try:
        prepare(root, out)
        raise AssertionError('기존 결과를 덮어썼습니다')
    except ValueError:
        pass
print('다중 시점 3D 중복, 누락 프레임, 표본 독립성, 출력 보호 검증')
