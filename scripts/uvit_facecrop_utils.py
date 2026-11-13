"""Conservative square crops matched to the released U-ViT source framing."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def artwork_landmarks(landmarks):
    """Map official artwork landmarks through its existing FFHQ-style quad."""
    left, right = [np.asarray(landmarks[key], float) for key in ['eye_left', 'eye_right']]
    mouth = (np.asarray(landmarks['mouth_left']) + np.asarray(landmarks['mouth_right'])) / 2
    eyes = (left + right) / 2
    delta = mouth - eyes
    x = right - left - np.flipud(delta) * [-1, 1]
    x /= np.linalg.norm(x)
    x *= max(np.linalg.norm(right - left) * 2, np.linalg.norm(delta) * 1.8)
    y = np.flipud(x) * [-1, 1]
    center = eyes + delta * .1
    origin = center - x - y
    basis = np.column_stack([2 * x, 2 * y])
    points = np.asarray([left, right, mouth])
    # Geometric quad coordinates; subpixel filtering in the existing export is retained.
    aligned = (points - origin) @ np.linalg.inv(basis).T * 128
    return aligned.tolist(), float(np.linalg.norm(x) * 2)


class AnimalEyes:
    """Official RTMPose AP-10K ONNX model; full-image top-down bbox on CPU."""
    def __init__(self, model):
        import cv2
        import onnxruntime as ort
        self.cv = cv2
        cv2.setNumThreads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        self.height, self.width = self.input.shape[-2:]

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        center = np.array([w / 2, h / 2])
        scale = np.array([w, h], float)
        ratio = self.width / self.height
        if scale[0] > scale[1] * ratio:
            scale[1] = scale[0] / ratio
        else:
            scale[0] = scale[1] * ratio
        scale *= 1.25
        factor = np.array([self.width, self.height]) / scale
        matrix = np.array([[factor[0], 0, self.width / 2 - center[0] * factor[0]],
                           [0, factor[1], self.height / 2 - center[1] * factor[1]]], np.float32)
        resized = self.cv.warpAffine(rgb, matrix, (self.width, self.height), flags=self.cv.INTER_LINEAR)
        # PIL gives RGB, matching the official SDK's BGR-to-RGB normalization.
        x = ((resized.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375])
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        sx, sy = self.session.run(None, {self.input.name: x})
        locs = np.stack([sx.argmax(-1), sy.argmax(-1)], -1)[0] / 2
        points = locs / [self.width, self.height] * scale + center - scale / 2
        scores = np.minimum(sx.max(-1), sy.max(-1))[0]
        return points[:3], scores[:3]

    def __call__(self, rgb):
        points, scores = self.predict(rgb)
        reverse, reverse_scores = self.predict(np.ascontiguousarray(rgb[:, ::-1]))
        reverse[:, 0] = rgb.shape[1] - 1 - reverse[:, 0]
        # AP-10K labels are animal-left/right; sort into image-left/right.
        order, reverse_order = np.argsort(points[:2, 0]), np.argsort(reverse[:2, 0])
        eyes, other = points[:2][order], reverse[:2][reverse_order]
        error = float(np.max(np.linalg.norm(eyes - other, axis=1)) / rgb.shape[1])
        confidence = float(min(scores[:2].min(), reverse_scores[:2].min()))
        average = (eyes + other) / 2
        gap = float(np.linalg.norm(average[1] - average[0]) / rgb.shape[1])
        flags = []
        if confidence < .6:
            flags.append('animal_low_eye_score')
        if error > .06:
            flags.append('animal_flip_disagreement')
        if not .10 <= gap <= .65:
            flags.append('animal_implausible_eye_distance')
        if np.max(np.abs(average[:, 1] - average[:, 1].mean())) > .10 * rgb.shape[0]:
            flags.append('animal_large_eye_tilt')
        if np.any(average < 0) or np.any(average > rgb.shape[1]):
            flags.append('animal_eye_outside_image')
        return average, dict(eye_score=confidence, flip_error_fraction=error,
            raw_points=points.tolist(), raw_scores=scores.tolist(), flags=flags)


def crop_box(points, template, extent=128., max_zoom=1.3, native_extent=128.):
    """Bounded isotropic zoom/translation; never rotate, pad, or enlarge native pixels."""
    points, template = np.asarray(points, float), np.asarray(template, float)
    assert points.shape == template.shape and len(points) in (2, 3)
    assert np.isfinite(points).all() and native_extent >= 64
    flags = []
    if len(points) == 3:
        weights = np.array([1., 1., .5])
    else:
        weights = np.ones(2)
    weights /= weights.sum()
    mean_p, mean_t = weights @ points, weights @ template
    def error(box):
        x, y, right, bottom = box
        return float(np.sqrt(np.sum(weights[:, None] * ((points - [x, y]) / (right - x) - template) ** 2)))
    original = [0., 0., extent, extent]
    best, best_error = original, error(original)
    cap = min(max_zoom, native_extent / 64, extent / 64)
    for zoom in np.linspace(1., cap, 61):
        side = extent / zoom
        origin = np.clip(mean_p - side * mean_t, 0, extent - side)
        box = [*origin, *(origin + side)]
        residual = error(box)
        if residual < best_error - 1e-10:
            best, best_error = [float(v) for v in box], residual
    zoom = extent / (best[2] - best[0])
    if best_error > .08:
        flags.append('remaining_landmark_mismatch')
    if abs(zoom - cap) < 1e-6 and zoom > 1.001:
        flags.append('zoom_at_conservative_limit')
    side = best[2] - best[0]
    assert side >= 64 and native_extent * side / extent >= 64 - 1e-7
    assert 0 <= best[0] <= best[2] <= extent and 0 <= best[1] <= best[3] <= extent
    assert best_error <= error(original) + 1e-9
    return dict(box=best, zoom=float(zoom), landmark_error_before=error(original),
        landmark_error_after=best_error, effective_native_extent=native_extent * side / extent,
        cropped=zoom > 1.001, flags=flags)


def render_crop(row, template, animal=None):
    with Image.open(row['input']) as image:
        assert image.mode == 'RGB' and image.size == (128, 128)
        rgb = np.array(image)
    audit = {}
    if row.get('landmarks') is None:
        assert animal is not None
        points, audit = animal(rgb)
        expected = np.asarray(template['anchors'])[:2]
    else:
        points = np.asarray(row['landmarks'])
        expected = np.asarray(template['anchors'])
    if audit.get('flags'):
        info = dict(box=[0., 0., 128., 128.], zoom=1., cropped=False,
                    flags=audit['flags'], effective_native_extent=row['native_extent'],
                    landmark_error_before=None, landmark_error_after=None)
    else:
        info = crop_box(points, expected, native_extent=row['native_extent'])
    info.update(landmarks=points.tolist(), detector=audit,
                needs_review=bool(audit.get('flags')) or info.get('landmark_error_after', 0) > .08)
    output = Image.fromarray(rgb).resize((64, 64), Image.Resampling.LANCZOS, box=tuple(info['box']))
    return output, info


def self_check():
    template = np.array([[.35, .42], [.65, .42], [.50, .74]])
    identity = crop_box(128 * template, template)
    assert identity['zoom'] == 1 and identity['landmark_error_after'] < 1e-12
    known = crop_box(100 * template + [14, 8], template)
    assert np.max(np.abs(np.array(known['box']) - [14, 8, 114, 108])) < .5
    assert known['landmark_error_after'] < known['landmark_error_before']
    for points in [90 * template - [10, 10], 160 * template + [5, 5]]:
        crop_box(points, template)
    assert not crop_box(128 * template, template)['cropped']
    print('CROP INVARIANTS PASSED: identity, known transform, boundary bounds, no upsampling, non-increasing landmark error', flush=True)
