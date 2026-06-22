import cv2
import numpy as np


# =========================================================
# READ IMAGE
# =========================================================
def read_image_from_bytes(image_bytes: bytes):
    if not image_bytes:
        raise ValueError("Bytes de imagen vacios")

    arr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("No se pudo decodificar imagen")

    return img


def read_upload_file(upload_file):
    data = upload_file.file.read()

    if not data:
        raise ValueError("Archivo vacio")

    return read_image_from_bytes(data)


# =========================================================
# IMAGE HELPERS
# =========================================================
def ensure_bgr(img: np.ndarray):
    if img is None:
        raise ValueError("Imagen invalida")

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img.copy()


def ensure_gray(img: np.ndarray):
    if img is None:
        raise ValueError("Imagen invalida")

    if len(img.shape) == 2:
        return img.copy()

    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =========================================================
# BBOX
# =========================================================
def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))

    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    return x1, y1, x2, y2


def crop_from_bbox(img, bbox):
    x1 = bbox["x1"]
    y1 = bbox["y1"]
    x2 = bbox["x2"]
    y2 = bbox["y2"]

    h, w = img.shape[:2]

    x1, y1, x2, y2 = clamp_bbox(
        x1,
        y1,
        x2,
        y2,
        w,
        h,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop.copy()


# =========================================================
# RESIZE
# =========================================================
def resize_with_scale(
    img: np.ndarray,
    scale: float,
):
    if scale == 1.0:
        return img.copy()

    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )