import cv2
import numpy as np

from config import (
    ALIGNED_WIDTH,
    ALIGNED_HEIGHT,
    MIN_COMPONENT_AREA,
)

from image_utils import ensure_gray


def normalize_foreground(mask):
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    nonzero = int(np.count_nonzero(mask))
    total = int(mask.shape[0] * mask.shape[1])

    if total > 0 and nonzero > total * 0.60:
        mask = cv2.bitwise_not(mask)

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def remove_text_like_components(binary_img):
    binary_img = normalize_foreground(binary_img)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary_img)

    for label_idx in range(1, num_labels):
        x = stats[label_idx, cv2.CC_STAT_LEFT]
        y = stats[label_idx, cv2.CC_STAT_TOP]
        w = stats[label_idx, cv2.CC_STAT_WIDTH]
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        area = stats[label_idx, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        aspect = w / max(h, 1)

        looks_like_text = (
            area < 90
            and h < 22
            and w < 45
        )

        looks_like_thin_line = (
            h <= 4
            and aspect > 12
        )

        if looks_like_text:
            continue

        if looks_like_thin_line:
            continue

        cleaned[labels == label_idx] = 255

    return normalize_foreground(cleaned)


def remove_small_components(binary_img, min_area=MIN_COMPONENT_AREA):
    binary_img = normalize_foreground(binary_img)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary_img)

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])

        if area >= min_area:
            cleaned[labels == label_idx] = 255

    return normalize_foreground(cleaned)


def crop_to_content(binary_img, padding=10):
    binary_img = normalize_foreground(binary_img)

    coords = cv2.findNonZero(binary_img)

    if coords is None:
        return binary_img.copy()

    x, y, w, h = cv2.boundingRect(coords)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(binary_img.shape[1], x + w + padding)
    y2 = min(binary_img.shape[0], y + h + padding)

    return binary_img[y1:y2, x1:x2].copy()


def binarize_signature(img):
    gray = ensure_gray(img)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0,
    )

    binary_adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )

    _, binary_otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    adaptive_pixels = int(np.count_nonzero(binary_adaptive))
    otsu_pixels = int(np.count_nonzero(binary_otsu))

    if otsu_pixels > 20 and otsu_pixels < adaptive_pixels * 0.75:
        binary = binary_otsu
    else:
        binary = binary_adaptive

    binary = normalize_foreground(binary)

    kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    binary = remove_small_components(binary)
    binary = remove_text_like_components(binary)

    return normalize_foreground(binary)


def resize_and_center(
    binary_img,
    width=ALIGNED_WIDTH,
    height=ALIGNED_HEIGHT,
):
    binary_img = normalize_foreground(binary_img)

    h, w = binary_img.shape[:2]

    if h <= 0 or w <= 0:
        return np.zeros((height, width), dtype=np.uint8)

    scale = min(width / w, height / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        binary_img,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros((height, width), dtype=np.uint8)

    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    return normalize_foreground(canvas)


def preprocess_signature(img):
    binary = binarize_signature(img)
    cropped = crop_to_content(binary, padding=8)
    aligned = resize_and_center(cropped)

    return {
        "binary": binary,
        "crop": cropped,
        "aligned": aligned,
    }