import cv2
import numpy as np
from ultralytics import YOLO

from config import YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD
from image_utils import clamp_bbox, resize_with_scale

model = YOLO(YOLO_MODEL_PATH)


def enhance_document(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def remove_small_components(binary, min_area=20):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary)

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == idx] = 255

    return cleaned


def tight_signature_crop(crop_bgr: np.ndarray):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )

    binary = remove_small_components(binary, min_area=20)

    coords = cv2.findNonZero(binary)

    if coords is None:
        return crop_bgr.copy()

    x, y, w, h = cv2.boundingRect(coords)

    padding = 12

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(crop_bgr.shape[1], x + w + padding)
    y2 = min(crop_bgr.shape[0], y + h + padding)

    final_crop = crop_bgr[y1:y2, x1:x2]

    if final_crop is None or final_crop.size == 0:
        return crop_bgr.copy()

    return final_crop.copy()


def bbox_iou(a: dict, b: dict):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def detect_yolo(img: np.ndarray):
    results = model.predict(
        source=img,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None:
        return []

    h, w = img.shape[:2]

    detections = []

    for box in result.boxes:
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = clamp_bbox(
            xyxy[0],
            xyxy[1],
            xyxy[2],
            xyxy[3],
            w,
            h,
        )

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            },
            "confidence": float(conf),
        })

    return detections


def deduplicate_detections(detections, iou_threshold=0.45):
    ordered = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    selected = []

    for det in ordered:
        duplicated = False

        for kept in selected:
            if bbox_iou(det["bbox"], kept["bbox"]) >= iou_threshold:
                duplicated = True
                break

        if not duplicated:
            selected.append(det)

    selected = sorted(
        selected,
        key=lambda item: (
            item["bbox"]["y1"],
            item["bbox"]["x1"],
        ),
    )

    return selected


def detect_signatures(img: np.ndarray):
    enhanced = enhance_document(img)

    variants = [
        ("original", img, 1.0),
        ("enhanced", enhanced, 1.0),
        ("original_x2", resize_with_scale(img, 2.0), 2.0),
        ("enhanced_x2", resize_with_scale(enhanced, 2.0), 2.0),
    ]

    h, w = img.shape[:2]

    all_detections = []

    for variant_name, variant_img, scale in variants:
        detections = detect_yolo(variant_img)

        for det in detections:
            bbox = det["bbox"]

            x1 = int(round(bbox["x1"] / scale))
            y1 = int(round(bbox["y1"] / scale))
            x2 = int(round(bbox["x2"] / scale))
            y2 = int(round(bbox["y2"] / scale))

            x1, y1, x2, y2 = clamp_bbox(
                x1,
                y1,
                x2,
                y2,
                w,
                h,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2].copy()

            if crop is None or crop.size == 0:
                continue

            tight_crop = tight_signature_crop(crop)

            all_detections.append({
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
                "confidence": round(float(det["confidence"]), 4),
                "source_variant": variant_name,
                "crop": tight_crop,
                "raw_crop": crop,
            })

    return deduplicate_detections(all_detections)