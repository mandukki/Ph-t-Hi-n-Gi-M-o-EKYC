"""Approximate edit boxes from the controlled synthetic-image generator.

These coordinates are weak annotations. They are scaled from the generator's
1600x1000 card and do not account for a later perspective warp.
"""

from pathlib import Path

TEXT_ROIS = [
    [(660, 470, 1100, 535)], [(750, 555, 1100, 615)],
    [(550, 620, 1250, 690)], [(750, 820, 1100, 880)],
    [(750, 690, 1300, 760)],
    [(660, 470, 1100, 535), (550, 620, 1250, 690)],
]
SPLICE_ROIS = [
    [(950, 650, 1170, 870)], [(1260, 700, 1480, 920)],
    [(720, 730, 1100, 850)], [(1100, 780, 1420, 910)],
    [(1300, 680, 1480, 860)], [(50, 50, 230, 230)],
]


def weak_rois(path: str) -> list[tuple[int, int, int, int]]:
    stem = Path(path).stem
    variant = int(stem.rsplit("_", 1)[1]) - 1
    if not 0 <= variant < 6:
        raise ValueError(f"Unsupported synthetic variant: {path}")
    if "_fake_photo_" in stem:
        rois = [(50, 430, 480, 930)]
    elif "_fake_text_" in stem:
        rois = TEXT_ROIS[variant]
    elif "_fake_splice_" in stem:
        rois = SPLICE_ROIS[variant]
    else:
        raise ValueError(f"No synthetic edit box for {path}")
    return [
        (round(x1 * 856 / 1600), round(y1 * 540 / 1000),
         round(x2 * 856 / 1600), round(y2 * 540 / 1000))
        for x1, y1, x2, y2 in rois
    ]


def intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
