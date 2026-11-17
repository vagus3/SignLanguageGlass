"""Split archives: numbered parts and aihubshell's byte-offset parts merge in the right order."""
import _path  # noqa: F401
import tempfile
import zipfile
from pathlib import Path

from tools.merge_parts import check, find_groups, merge, verify_zip

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    with zipfile.ZipFile(td / "source.zip", "w") as z:
        for i in range(40):
            z.writestr(f"clip/{i:03d}.json", str(i) * 500)
    data = (td / "source.zip").read_bytes()
    step = len(data) // 11 + 1
    chunks = [data[i:i + step] for i in range(0, len(data), step)]

    numbered = td / "numbered"
    numbered.mkdir()
    for i, chunk in enumerate(chunks, 1):          # part10/part11 sort before part2 as text
        (numbered / f"a.zip.part{i}").write_bytes(chunk)
    (target, parts), = find_groups(numbered).items()
    assert check(parts) == ([], False, 1, len(chunks))
    merge(target, parts, verbose=False)
    assert target.read_bytes() == data and verify_zip(target).startswith("정상")
    assert check(parts[:3] + parts[4:])[0] == [4]

    offsets = td / "offsets"                       # aihubshell: part<start byte>
    offsets.mkdir()
    big = 5_000_000_000                            # a contiguous-number check would need 5e9 ints
    position = 0
    for chunk in chunks:
        (offsets / f"b.zip.part{position}").write_bytes(chunk)
        position += len(chunk)
    (target, parts), = find_groups(offsets).items()
    assert check(parts)[0] == []
    merge(target, parts, verbose=False)
    assert target.read_bytes() == data
    assert check(parts[:5] + parts[6:])[0] == [parts[5][0]], "a missing piece is named"
    (offsets / f"c.zip.part0").write_bytes(b"x" * 10)
    (offsets / f"c.zip.part{big}").write_bytes(b"y")
    lost = find_groups(offsets)[offsets / "c.zip"]
    assert check(lost)[0] == [10], "gap detected without enumerating every number"
print("[OK] merge parts: numeric order, byte-offset names, missing pieces")
