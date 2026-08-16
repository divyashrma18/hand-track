"""Smoke test for hand_frame_effect.py — no webcam needed.

Verifies:
  * all three style filters run on a synthetic frame without error and
    return correctly-shaped BGR images
  * get_frame_rect() computes a sane rectangle from mock hand landmarks,
    and correctly returns None when there's not enough data
"""

import numpy as np

import hand_frame_effect as hfe


class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeHandLandmarks:
    def __init__(self, points):
        # points: dict {index: (x, y)} in normalized [0,1] coords
        self.landmark = [FakeLandmark(0.5, 0.5) for _ in range(21)]
        for idx, (x, y) in points.items():
            self.landmark[idx] = FakeLandmark(x, y)


def test_filters():
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    for name, fn in zip(hfe.FILTER_NAMES, hfe.FILTERS):
        out = fn(frame, 1) if fn is hfe.cartoon_filter else fn(frame)
        assert out.shape == frame.shape, f"{name}: shape mismatch {out.shape} vs {frame.shape}"
        assert out.dtype == np.uint8, f"{name}: dtype mismatch {out.dtype}"
        print(f"  [ok] {name} filter -> {out.shape}")


def test_frame_rect_two_hands():
    w, h = 1280, 720
    left_hand = FakeHandLandmarks({4: (0.2, 0.3), 8: (0.25, 0.35)})
    right_hand = FakeHandLandmarks({4: (0.7, 0.6), 8: (0.75, 0.65)})
    rect = hfe.get_frame_rect([left_hand, right_hand], w, h, margin=10)
    assert rect is not None, "expected a rectangle from two hands"
    x1, y1, x2, y2 = rect
    assert 0 <= x1 < x2 <= w
    assert 0 <= y1 < y2 <= h
    print(f"  [ok] two-hand rect -> {rect}")


def test_frame_rect_one_hand_returns_none():
    w, h = 1280, 720
    only_hand = FakeHandLandmarks({4: (0.2, 0.3), 8: (0.25, 0.35)})
    rect = hfe.get_frame_rect([only_hand], w, h)
    assert rect is None, "one hand shouldn't be enough to form a frame"
    print("  [ok] single-hand input correctly returns None")


def test_frame_rect_too_small_returns_none():
    w, h = 1280, 720
    left_hand = FakeHandLandmarks({4: (0.50, 0.50), 8: (0.501, 0.501)})
    right_hand = FakeHandLandmarks({4: (0.502, 0.502), 8: (0.503, 0.503)})
    rect = hfe.get_frame_rect([left_hand, right_hand], w, h, margin=0)
    assert rect is None, "a near-zero-area rect should be rejected"
    print("  [ok] degenerate tiny rect correctly returns None")


if __name__ == "__main__":
    print("Testing style filters...")
    test_filters()
    print("Testing get_frame_rect...")
    test_frame_rect_two_hands()
    test_frame_rect_one_hand_returns_none()
    test_frame_rect_too_small_returns_none()
    print("\nAll smoke tests passed.")
