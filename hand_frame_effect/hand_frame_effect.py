"""
hand_frame_effect.py

Recreates the viral "hand-frame reveal" effect:
Form a rectangle/frame with your hands (thumb + index finger of each hand)
in front of your webcam, and the region INSIDE that frame shows a
stylized "alternate you" (cartoon / anime-ish / Ghibli-warm / sketch),
while everything OUTSIDE the frame stays as the normal camera feed.

Dependencies:
    pip install opencv-python mediapipe numpy

Run:
    python hand_frame_effect.py

Controls:
    q       - quit
    s       - cycle through style filters (cartoon / ghibli-warm / sketch)
    d       - toggle hand-landmark debug overlay
    +/-     - adjust cartoon edge strength
"""

import time

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Stylization filters
# ---------------------------------------------------------------------------
# These are fast, dependency-free "anime/Ghibli-ish" looks built from classic
# CV techniques (bilateral filtering + edge overlay + color grading), so they
# run in real time on a CPU with no model download required. If you want a
# truer neural anime/Ghibli look, see "Swapping in a real AI style model"
# at the bottom of this file — stylize() is the single plug point.


def cartoon_filter(frame, edge_strength=1):
    """Classic cartoon look: smooth flat color regions + bold dark edges."""
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    color = small
    for _ in range(2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
    color = cv2.resize(color, (frame.shape[1], frame.shape[0]))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=9, C=2 + edge_strength,
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    cartoon = cv2.bitwise_and(color, edges)

    hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.35, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.05, 0, 255)
    cartoon = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cartoon


def ghibli_warm_filter(frame):
    """Warmer, softer, slightly painterly grade reminiscent of Ghibli scenes."""
    smooth = cv2.edgePreservingFilter(frame, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)

    b, g, r = cv2.split(smooth.astype(np.float32))
    r = np.clip(r * 1.08 + 8, 0, 255)
    g = np.clip(g * 1.03 + 4, 0, 255)
    b = np.clip(b * 0.94, 0, 255)
    graded = cv2.merge([b, g, r]).astype(np.uint8)

    blur = cv2.GaussianBlur(graded, (0, 0), sigmaX=8)
    glow = cv2.addWeighted(graded, 0.75, blur, 0.25, 0)
    return glow


def sketch_filter(frame):
    """Pencil-sketch style — another common 'alternate you' look."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


FILTERS = [cartoon_filter, ghibli_warm_filter, sketch_filter]
FILTER_NAMES = ["cartoon", "ghibli-warm", "sketch"]


def stylize(frame, filter_index, edge_strength):
    """Single plug point — swap this out for a real neural style model."""
    fn = FILTERS[filter_index]
    if fn is cartoon_filter:
        return fn(frame, edge_strength)
    return fn(frame)


# ---------------------------------------------------------------------------
# Hand-frame detection
# ---------------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_frame_rect(hand_landmarks_list, frame_w, frame_h, margin=10):
    """
    Given MediaPipe landmarks for one or two hands, compute the rectangle
    formed by the thumb tip (landmark 4) and index-finger tip (landmark 8)
    of each hand. Returns (x1, y1, x2, y2) in pixel coords, or None if we
    don't have enough points to form a sensible rectangle.
    """
    pts = []
    for hand_landmarks in hand_landmarks_list:
        for idx in (4, 8):  # thumb tip, index tip
            lm = hand_landmarks.landmark[idx]
            pts.append((lm.x * frame_w, lm.y * frame_h))

    if len(pts) < 4:  # need both hands for a clean rectangle
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = int(min(xs)) - margin, int(max(xs)) + margin
    y1, y2 = int(min(ys)) - margin, int(max(ys)) + margin

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)

    if x2 - x1 < 40 or y2 - y1 < 40:  # too small to be a real "frame"
        return None
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    filter_index = 0
    edge_strength = 1
    show_debug = False
    prev_time = time.time()

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for a natural "selfie" feel
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display = frame.copy()

            if results.multi_hand_landmarks:
                rect = get_frame_rect(results.multi_hand_landmarks, w, h)

                if rect:
                    x1, y1, x2, y2 = rect
                    styled_full = stylize(frame, filter_index, edge_strength)
                    display[y1:y2, x1:x2] = styled_full[y1:y2, x1:x2]

                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 3)
                    cv2.putText(
                        display, FILTER_NAMES[filter_index], (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
                    )

                if show_debug:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 200, 0), thickness=1),
                        )

            now = time.time()
            fps = 1 / (now - prev_time) if now != prev_time else 0
            prev_time = now
            cv2.putText(display, f"{fps:.0f} FPS  |  [s] style  [d] debug  [q] quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Hand-Frame Reveal Effect", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filter_index = (filter_index + 1) % len(FILTERS)
            elif key == ord('d'):
                show_debug = not show_debug
            elif key in (ord('+'), ord('=')):
                edge_strength = min(edge_strength + 1, 5)
            elif key in (ord('-'), ord('_')):
                edge_strength = max(edge_strength - 1, 0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Swapping in a real AI style model (true anime / Ghibli look)
# ---------------------------------------------------------------------------
# The filters above are classic-CV approximations that run at full webcam
# frame rate with zero setup. For a more authentic anime/Ghibli look, swap
# stylize() to call a pretrained model instead, e.g.:
#
#   1. AnimeGANv2 / AnimeGAN (ONNX export) run via onnxruntime — still close
#      to real-time on a decent CPU/GPU, purpose-built for this look.
#      pip install onnxruntime, then load the .onnx model once at startup
#      and run inference only on the cropped [y1:y2, x1:x2] region (much
#      faster than the whole frame).
#
#   2. A Stable Diffusion img2img call (e.g. a "Ghibli style" LoRA) gives
#      the best quality but is too slow for live video (think ~1-3 sec per
#      frame) — better suited to stylizing a captured photo/short clip
#      after the fact rather than a live webcam loop.
#
# Either way, only replace the body of stylize(); get_frame_rect() and the
# main loop's compositing logic (display[y1:y2, x1:x2] = styled[...]) don't
# need to change.
