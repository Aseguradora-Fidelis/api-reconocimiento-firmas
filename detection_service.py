from datetime import datetime
import logging
from pathlib import Path
import re

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CAMERA_SIGNATURE_MERGE_GAP_RATIO,
    CAMERA_SIGNATURE_MERGE_OVERLAP,
    CAMERA_SIGNATURE_PADDING_X,
    CAMERA_SIGNATURE_PADDING_Y,
    DEBUG_SAVE_FILES,
    SIGNATURE_BEST_CONFIDENCE_RATIO,
    SIGNATURE_DETECTION_DEBUG,
    SIGNATURE_DETECTION_DEBUG_DIR,
    SIGNATURE_DEDUP_CONTAINMENT_THRESHOLD,
    SIGNATURE_DEDUP_IOU_THRESHOLD,
    SIGNATURE_MAX_VARIANT_PIXELS,
    SIGNATURE_MIN_KEEP_CONFIDENCE,
    SIGNATURE_TILE_COLS,
    SIGNATURE_TILE_MIN_DIM,
    SIGNATURE_TILE_OVERLAP,
    SIGNATURE_TILE_ROWS,
    YOLO_CONF_THRESHOLD,
    YOLO_IMGSZ,
    YOLO_IOU_THRESHOLD,
    YOLO_MAX_DET,
    YOLO_MODEL_PATH,
)
from image_utils import clamp_bbox, resize_with_scale

logger = logging.getLogger(__name__)

model = YOLO(YOLO_MODEL_PATH)


def context_label(debug_context=None):
    if not debug_context:
        return "sin_contexto"

    parts = []

    for key in (
        "source",
        "archivo",
        "s3_key",
        "page",
    ):
        value = debug_context.get(key)

        if value is not None:
            parts.append(f"{key}={value}")

    return " ".join(parts) if parts else "sin_contexto"


def safe_file_part(value):
    value = str(value or "page")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    return value[:120] or "page"


def build_predict_kwargs(source):
    kwargs = {
        "source": source,
        "conf": YOLO_CONF_THRESHOLD,
        "iou": YOLO_IOU_THRESHOLD,
        "max_det": YOLO_MAX_DET,
        "verbose": False,
    }

    if YOLO_IMGSZ and YOLO_IMGSZ > 0:
        kwargs["imgsz"] = YOLO_IMGSZ

    return kwargs


def model_size_from_error(exc):
    match = re.search(
        r"max model size\s*\(1,\s*3,\s*(\d+),\s*(\d+)\)",
        str(exc),
    )

    if not match:
        return None

    height = int(match.group(1))
    width = int(match.group(2))

    if height != width:
        return None

    return height


def build_predict_attempts(source):
    attempts = [
        build_predict_kwargs(source),
    ]

    unique_attempts = []
    seen = set()

    for kwargs in attempts:
        key = tuple(sorted(kwargs.keys()))

        if key in seen:
            continue

        seen.add(key)
        unique_attempts.append(kwargs)

    return unique_attempts


def predict_yolo(source, debug_context=None, variant_name="original"):
    last_exc = None
    attempted_sizes = set()

    for kwargs in build_predict_attempts(source):
        attempted_sizes.add(kwargs.get("imgsz"))

        try:
            return model.predict(**kwargs)
        except Exception as exc:
            last_exc = exc
            model_imgsz = model_size_from_error(exc)
            logger.warning(
                "YOLO fallo; reintentando si hay fallback context=%s "
                "variant=%s kwargs=%s error=%s",
                context_label(debug_context),
                variant_name,
                {
                    key: value
                    for key, value in kwargs.items()
                    if key != "source"
                },
                exc,
            )

            if model_imgsz and model_imgsz not in attempted_sizes:
                retry_kwargs = {
                    **kwargs,
                    "imgsz": model_imgsz,
                }
                attempted_sizes.add(model_imgsz)

                try:
                    logger.warning(
                        "Reintentando YOLO con imgsz fijo del engine=%s "
                        "context=%s variant=%s",
                        model_imgsz,
                        context_label(debug_context),
                        variant_name,
                    )
                    return model.predict(**retry_kwargs)
                except Exception as retry_exc:
                    last_exc = retry_exc
                    logger.warning(
                        "YOLO fallo con imgsz fijo del engine context=%s "
                        "variant=%s imgsz=%s error=%s",
                        context_label(debug_context),
                        variant_name,
                        model_imgsz,
                        retry_exc,
                    )

    raise RuntimeError(
        "YOLO no pudo ejecutar inferencia para "
        f"{context_label(debug_context)} variant={variant_name}: {last_exc}"
    ) from last_exc


def should_include_variant(img, scale):
    height, width = img.shape[:2]
    scaled_pixels = int(height * scale) * int(width * scale)

    return scaled_pixels <= SIGNATURE_MAX_VARIANT_PIXELS


def tile_bounds(length, parts, overlap_ratio):
    if parts <= 1:
        return [(0, length)]

    base = int(np.ceil(length / parts))
    overlap = int(round(base * overlap_ratio))
    bounds = []

    for index in range(parts):
        start = max(0, (index * base) - overlap)
        end = min(length, ((index + 1) * base) + overlap)

        if end <= start:
            continue

        bounds.append((start, end))

    return bounds


def iter_tiles(img):
    height, width = img.shape[:2]

    if max(height, width) < SIGNATURE_TILE_MIN_DIM:
        return []

    x_bounds = tile_bounds(
        width,
        SIGNATURE_TILE_COLS,
        SIGNATURE_TILE_OVERLAP,
    )
    y_bounds = tile_bounds(
        height,
        SIGNATURE_TILE_ROWS,
        SIGNATURE_TILE_OVERLAP,
    )

    tiles = []

    for row_index, (y1, y2) in enumerate(
        y_bounds,
        start=1,
    ):
        for col_index, (x1, x2) in enumerate(
            x_bounds,
            start=1,
        ):
            tile = img[y1:y2, x1:x2]

            if tile is None or tile.size == 0:
                continue

            tiles.append({
                "name": f"tile_r{row_index}_c{col_index}",
                "img": tile.copy(),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            })

    return tiles


def build_detection_variants(img, enhanced):
    variants = [
        {
            "name": "original",
            "img": img,
            "scale": 1.0,
            "offset_x": 0,
            "offset_y": 0,
        },
        {
            "name": "enhanced",
            "img": enhanced,
            "scale": 1.0,
            "offset_x": 0,
            "offset_y": 0,
        },
    ]

    for base_name, source_img in (
        ("original", img),
        ("enhanced", enhanced),
    ):
        for tile in iter_tiles(source_img):
            variants.append({
                "name": f"{base_name}_{tile['name']}",
                "img": tile["img"],
                "scale": 1.0,
                "offset_x": tile["x1"],
                "offset_y": tile["y1"],
                "tile_bbox": {
                    "x1": tile["x1"],
                    "y1": tile["y1"],
                    "x2": tile["x2"],
                    "y2": tile["y2"],
                },
            })

    for base_name, source_img in (
        ("original_x2", img),
        ("enhanced_x2", enhanced),
    ):
        if (
            max(source_img.shape[:2]) >= SIGNATURE_TILE_MIN_DIM
            or not should_include_variant(source_img, 2.0)
        ):
            logger.info(
                "Saltando variante x2 variant=%s size=%sx%s "
                "min_dim=%s max_pixels=%s",
                base_name,
                source_img.shape[1],
                source_img.shape[0],
                SIGNATURE_TILE_MIN_DIM,
                SIGNATURE_MAX_VARIANT_PIXELS,
            )
            continue

        variants.append({
            "name": base_name,
            "img": resize_with_scale(source_img, 2.0),
            "scale": 2.0,
            "offset_x": 0,
            "offset_y": 0,
        })

    return variants


def build_variant_error_debug(variant_name, variant_img, scale, exc):
    return {
        "variant": variant_name,
        "scale": scale,
        "image_width": int(variant_img.shape[1]),
        "image_height": int(variant_img.shape[0]),
        "raw_count": 0,
        "accepted_count": 0,
        "raw_boxes": [],
        "error": str(exc),
    }


def empty_yolo_debug(variant_name, error=None):
    debug = {
        "variant": variant_name,
        "raw_count": 0,
        "accepted_count": 0,
        "raw_boxes": [],
    }

    if error:
        debug["error"] = str(error)

    return debug


def log_variant_failure(debug_context, variant_name, exc):
    logger.exception(
        "Error ejecutando deteccion YOLO context=%s variant=%s error=%s",
            context_label(debug_context),
            variant_name,
            exc,
    )


def box_class_info(result, box):
    class_id = None
    class_name = None

    if getattr(box, "cls", None) is not None and len(box.cls) > 0:
        class_id = int(box.cls[0].item())

    names = getattr(result, "names", None)

    if names is not None and class_id is not None:
        if isinstance(names, dict):
            class_name = names.get(class_id)
        elif 0 <= class_id < len(names):
            class_name = names[class_id]

    return class_id, class_name


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


def bbox_intersection_ratio(a: dict, b: dict):
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
    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0

    return float(inter / smaller_area)


def detection_duplicate_reason(det, kept, iou_threshold, containment_threshold):
    iou = bbox_iou(det["bbox"], kept["bbox"])

    if iou >= iou_threshold:
        return "iou", iou, None

    containment = bbox_intersection_ratio(det["bbox"], kept["bbox"])

    if containment >= containment_threshold:
        return "containment", iou, containment

    return None, iou, containment


def axis_overlap_ratio(a1, a2, b1, b2):
    overlap = max(0, min(a2, b2) - max(a1, b1))
    smaller = min(max(0, a2 - a1), max(0, b2 - b1))

    if smaller <= 0:
        return 0.0

    return float(overlap / smaller)


def camera_boxes_belong_together(a: dict, b: dict):
    horizontal_overlap = axis_overlap_ratio(
        a["x1"], a["x2"], b["x1"], b["x2"]
    )
    vertical_overlap = axis_overlap_ratio(
        a["y1"], a["y2"], b["y1"], b["y2"]
    )

    if (
        horizontal_overlap >= CAMERA_SIGNATURE_MERGE_OVERLAP
        and vertical_overlap >= CAMERA_SIGNATURE_MERGE_OVERLAP
    ):
        return True

    horizontal_gap = max(a["x1"], b["x1"]) - min(a["x2"], b["x2"])
    vertical_gap = max(a["y1"], b["y1"]) - min(a["y2"], b["y2"])
    max_width = max(a["x2"] - a["x1"], b["x2"] - b["x1"])
    max_height = max(a["y2"] - a["y1"], b["y2"] - b["y1"])

    horizontally_close = horizontal_gap <= (
        max_width * CAMERA_SIGNATURE_MERGE_GAP_RATIO
    )
    vertically_close = vertical_gap <= (
        max_height * CAMERA_SIGNATURE_MERGE_GAP_RATIO
    )

    return (
        vertical_overlap >= CAMERA_SIGNATURE_MERGE_OVERLAP
        and horizontally_close
    ) or (
        horizontal_overlap >= CAMERA_SIGNATURE_MERGE_OVERLAP
        and vertically_close
    )


def consolidate_camera_signature(img, detections):
    """Une las cajas conectadas a la mejor deteccion de la camara."""
    if not detections:
        return [], 0

    anchor = max(
        detections,
        key=lambda item: float(item.get("confidence") or 0.0),
    )
    cluster = [anchor]
    pending = [det for det in detections if det is not anchor]

    changed = True

    while changed:
        changed = False

        for det in pending[:]:
            if any(
                camera_boxes_belong_together(det["bbox"], item["bbox"])
                for item in cluster
            ):
                cluster.append(det)
                pending.remove(det)
                changed = True

    union_x1 = min(item["bbox"]["x1"] for item in cluster)
    union_y1 = min(item["bbox"]["y1"] for item in cluster)
    union_x2 = max(item["bbox"]["x2"] for item in cluster)
    union_y2 = max(item["bbox"]["y2"] for item in cluster)
    union_width = union_x2 - union_x1
    union_height = union_y2 - union_y1
    pad_x = int(round(union_width * CAMERA_SIGNATURE_PADDING_X))
    pad_y = int(round(union_height * CAMERA_SIGNATURE_PADDING_Y))
    height, width = img.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(
        union_x1 - pad_x,
        union_y1 - pad_y,
        union_x2 + pad_x,
        union_y2 + pad_y,
        width,
        height,
    )
    crop = img[y1:y2, x1:x2].copy()

    if crop is None or crop.size == 0:
        return [anchor], len(cluster)

    merged = {
        "bbox": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
        },
        "confidence": round(
            max(float(item.get("confidence") or 0.0) for item in cluster),
            4,
        ),
        "class_id": anchor.get("class_id"),
        "class_name": anchor.get("class_name"),
        "source_variant": "camera_merged",
        # Para camara no se aplica tight_signature_crop: el preprocesamiento
        # del comparador ya recorta el contenido y aqui debemos conservar
        # todos los trazos detectados.
        "crop": crop,
        "raw_crop": crop.copy(),
        "merged_detection_count": len(cluster),
        "merged_boxes": [item["bbox"].copy() for item in cluster],
    }

    return [merged], len(cluster)


def detect_yolo(
    img: np.ndarray,
    variant_name="original",
    debug_context=None,
):
    results = predict_yolo(
        img,
        debug_context=debug_context,
        variant_name=variant_name,
    )

    if not results:
        return [], empty_yolo_debug(variant_name)

    result = results[0]

    if result.boxes is None:
        return [], empty_yolo_debug(variant_name)

    h, w = img.shape[:2]

    detections = []
    raw_boxes = []

    logger.info(
        "YOLO raw detections context=%s variant=%s image=%sx%s "
        "conf=%.4f iou=%.4f max_det=%s imgsz=%s count=%s",
        context_label(debug_context),
        variant_name,
        w,
        h,
        YOLO_CONF_THRESHOLD,
        YOLO_IOU_THRESHOLD,
        YOLO_MAX_DET,
        YOLO_IMGSZ,
        len(result.boxes),
    )

    for raw_index, box in enumerate(
        result.boxes,
        start=1,
    ):
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy()
        class_id, class_name = box_class_info(
            result,
            box,
        )

        raw_x1 = float(xyxy[0])
        raw_y1 = float(xyxy[1])
        raw_x2 = float(xyxy[2])
        raw_y2 = float(xyxy[3])
        raw_width = max(0.0, raw_x2 - raw_x1)
        raw_height = max(0.0, raw_y2 - raw_y1)

        raw_info = {
            "index": raw_index,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "bbox": {
                "x1": round(raw_x1, 2),
                "y1": round(raw_y1, 2),
                "x2": round(raw_x2, 2),
                "y2": round(raw_y2, 2),
            },
            "width": round(raw_width, 2),
            "height": round(raw_height, 2),
        }
        raw_boxes.append(raw_info)

        logger.info(
            "YOLO raw box context=%s variant=%s index=%s class=%s:%s "
            "conf=%.4f bbox=(%.1f,%.1f,%.1f,%.1f) size=%.1fx%.1f",
            context_label(debug_context),
            variant_name,
            raw_index,
            class_id,
            class_name,
            conf,
            raw_x1,
            raw_y1,
            raw_x2,
            raw_y2,
            raw_width,
            raw_height,
        )

        x1, y1, x2, y2 = clamp_bbox(
            raw_x1,
            raw_y1,
            raw_x2,
            raw_y2,
            w,
            h,
        )

        if x2 <= x1 or y2 <= y1:
            logger.info(
                "YOLO box descartada por bbox invalido context=%s "
                "variant=%s index=%s",
                context_label(debug_context),
                variant_name,
                raw_index,
            )
            continue

        detections.append({
            "bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            },
            "confidence": float(conf),
            "class_id": class_id,
            "class_name": class_name,
            "raw_index": raw_index,
        })

    return detections, {
        "variant": variant_name,
        "raw_count": len(raw_boxes),
        "accepted_count": len(detections),
        "raw_boxes": raw_boxes,
    }


def deduplicate_detections(
    detections,
    iou_threshold=SIGNATURE_DEDUP_IOU_THRESHOLD,
    containment_threshold=SIGNATURE_DEDUP_CONTAINMENT_THRESHOLD,
    debug_context=None,
):
    ordered = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    selected = []

    for det in ordered:
        duplicated = False

        for kept in selected:
            reason, iou, containment = detection_duplicate_reason(
                det,
                kept,
                iou_threshold,
                containment_threshold,
            )

            if reason:
                duplicated = True
                logger.info(
                    "Deteccion descartada por NMS context=%s reason=%s "
                    "iou=%.4f containment=%s iou_threshold=%.2f "
                    "containment_threshold=%.2f bbox=%s kept_bbox=%s",
                    context_label(debug_context),
                    reason,
                    iou,
                    (
                        round(containment, 4)
                        if containment is not None
                        else None
                    ),
                    iou_threshold,
                    containment_threshold,
                    det["bbox"],
                    kept["bbox"],
                )
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


def filter_best_confidence_detections(
    detections,
    ratio=SIGNATURE_BEST_CONFIDENCE_RATIO,
    min_keep_confidence=SIGNATURE_MIN_KEEP_CONFIDENCE,
    debug_context=None,
):
    if not detections:
        return [], None, 0

    best_confidence = max(
        float(det.get("confidence") or 0.0)
        for det in detections
    )
    relative_min_confidence = (
        best_confidence * ratio
        if ratio is not None and ratio > 0
        else 0.0
    )
    min_confidence = max(
        relative_min_confidence,
        min_keep_confidence or 0.0,
    )
    kept = []
    dropped_count = 0

    for det in detections:
        confidence = float(det.get("confidence") or 0.0)

        if confidence >= min_confidence:
            kept.append(det)
            continue

        dropped_count += 1
        logger.info(
            "Deteccion descartada por confianza final context=%s "
            "confidence=%.4f best_confidence=%.4f min_confidence=%.4f "
            "relative_min_confidence=%.4f min_keep_confidence=%.4f "
            "ratio=%.2f bbox=%s",
            context_label(debug_context),
            confidence,
            best_confidence,
            min_confidence,
            relative_min_confidence,
            min_keep_confidence or 0.0,
            ratio,
            det["bbox"],
        )

    return kept, min_confidence, dropped_count


def save_detection_debug_image(
    img,
    pre_nms_detections,
    selected_detections,
    debug_context=None,
):
    if not DEBUG_SAVE_FILES or not SIGNATURE_DETECTION_DEBUG:
        return None

    if not selected_detections:
        logger.info(
            "Imagen debug deteccion omitida sin firmas fuertes context=%s",
            context_label(debug_context),
        )
        return None

    output_dir = Path(SIGNATURE_DETECTION_DEBUG_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label = (
        debug_context.get("archivo")
        if debug_context
        else None
    ) or (
        debug_context.get("s3_key")
        if debug_context
        else None
    ) or (
        debug_context.get("source")
        if debug_context
        else None
    ) or "page"

    page = (
        debug_context.get("page")
        if debug_context
        else None
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"{timestamp}_{safe_file_part(label)}"
        f"_p{safe_file_part(page)}.jpg"
    )
    output_path = output_dir / filename

    canvas = img.copy()

    for det in pre_nms_detections:
        bbox = det["bbox"]
        cv2.rectangle(
            canvas,
            (bbox["x1"], bbox["y1"]),
            (bbox["x2"], bbox["y2"]),
            (0, 165, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"{det.get('source_variant')} {det['confidence']:.2f}",
            (bbox["x1"], max(16, bbox["y1"] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    for idx, det in enumerate(
        selected_detections,
        start=1,
    ):
        bbox = det["bbox"]
        cv2.rectangle(
            canvas,
            (bbox["x1"], bbox["y1"]),
            (bbox["x2"], bbox["y2"]),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            canvas,
            f"KEEP {idx} {det['confidence']:.2f}",
            (bbox["x1"], min(canvas.shape[0] - 8, bbox["y2"] + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(output_path),
        canvas,
    )

    logger.info(
        "Imagen debug deteccion guardada context=%s path=%s",
        context_label(debug_context),
        output_path,
    )

    return str(output_path)


def save_camera_original_debug_image(
    img,
    pre_nms_detections,
    selected_detections,
    debug_context=None,
):
    """Guarda una vista limpia de lo reconocido sobre la foto original."""
    if (
        not DEBUG_SAVE_FILES
        or not SIGNATURE_DETECTION_DEBUG
        or not debug_context
        or debug_context.get("source") != "camera"
    ):
        return None

    output_dir = Path(SIGNATURE_DETECTION_DEBUG_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / (
        f"{timestamp}_camera_original_recognized.jpg"
    )
    canvas = img.copy()
    original_detections = [
        det
        for det in pre_nms_detections
        if det.get("source_variant") == "original"
    ]

    for idx, det in enumerate(original_detections, start=1):
        bbox = det["bbox"]
        cv2.rectangle(
            canvas,
            (bbox["x1"], bbox["y1"]),
            (bbox["x2"], bbox["y2"]),
            (0, 165, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"ORIGINAL {idx} {det['confidence']:.2f}",
            (bbox["x1"], max(18, bbox["y1"] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 125, 255),
            2,
            cv2.LINE_AA,
        )

    for idx, det in enumerate(selected_detections, start=1):
        bbox = det["bbox"]
        cv2.rectangle(
            canvas,
            (bbox["x1"], bbox["y1"]),
            (bbox["x2"], bbox["y2"]),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            canvas,
            f"FIRMA COMPLETA {idx}",
            (bbox["x1"], min(canvas.shape[0] - 8, bbox["y2"] + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 160, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)
    logger.info(
        "Debug de camara original guardado context=%s path=%s "
        "original_count=%s final_count=%s",
        context_label(debug_context),
        output_path,
        len(original_detections),
        len(selected_detections),
    )

    return str(output_path)


def summarize_final_detection(det):
    bbox = det["bbox"]

    return {
        "bbox": bbox,
        "confidence": det["confidence"],
        "class_id": det.get("class_id"),
        "class_name": det.get("class_name"),
        "source_variant": det.get("source_variant"),
        "width": bbox["x2"] - bbox["x1"],
        "height": bbox["y2"] - bbox["y1"],
        "merged_detection_count": det.get("merged_detection_count", 1),
        "merged_boxes": det.get("merged_boxes"),
    }


def detect_signatures(
    img: np.ndarray,
    debug_context=None,
    return_debug=False,
):
    enhanced = enhance_document(img)

    variants = build_detection_variants(
        img,
        enhanced,
    )

    h, w = img.shape[:2]

    all_detections = []
    variants_debug = []

    for variant in variants:
        variant_name = variant["name"]
        variant_img = variant["img"]
        scale = variant["scale"]
        offset_x = variant.get("offset_x", 0)
        offset_y = variant.get("offset_y", 0)

        try:
            detections, variant_debug = detect_yolo(
                variant_img,
                variant_name=variant_name,
                debug_context=debug_context,
            )
        except Exception as exc:
            log_variant_failure(
                debug_context,
                variant_name,
                exc,
            )
            variants_debug.append(
                build_variant_error_debug(
                    variant_name,
                    variant_img,
                    scale,
                    exc,
                )
            )
            continue

        variant_debug["scale"] = scale
        variant_debug["image_width"] = int(variant_img.shape[1])
        variant_debug["image_height"] = int(variant_img.shape[0])
        variant_debug["offset_x"] = int(offset_x)
        variant_debug["offset_y"] = int(offset_y)

        if variant.get("tile_bbox"):
            variant_debug["tile_bbox"] = variant["tile_bbox"]

        variants_debug.append(variant_debug)

        for det in detections:
            bbox = det["bbox"]

            x1 = int(round(bbox["x1"] / scale)) + offset_x
            y1 = int(round(bbox["y1"] / scale)) + offset_y
            x2 = int(round(bbox["x2"] / scale)) + offset_x
            y2 = int(round(bbox["y2"] / scale)) + offset_y

            x1, y1, x2, y2 = clamp_bbox(
                x1,
                y1,
                x2,
                y2,
                w,
                h,
            )

            if x2 <= x1 or y2 <= y1:
                logger.info(
                    "Deteccion descartada al reescalar context=%s "
                    "variant=%s bbox=%s scale=%s",
                    context_label(debug_context),
                    variant_name,
                    bbox,
                    scale,
                )
                continue

            crop = img[y1:y2, x1:x2].copy()

            if crop is None or crop.size == 0:
                logger.info(
                    "Deteccion descartada por crop vacio context=%s "
                    "variant=%s bbox=%s",
                    context_label(debug_context),
                    variant_name,
                    bbox,
                )
                continue

            tight_crop = tight_signature_crop(crop)
            final_bbox = {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }

            all_detections.append({
                "bbox": final_bbox,
                "confidence": round(float(det["confidence"]), 4),
                "class_id": det.get("class_id"),
                "class_name": det.get("class_name"),
                "source_variant": variant_name,
                "crop": tight_crop,
                "raw_crop": crop,
            })

    selected = deduplicate_detections(
        all_detections,
        debug_context=debug_context,
    )
    post_nms_count = len(selected)
    is_camera = bool(
        debug_context and debug_context.get("source") == "camera"
    )
    camera_merged_detection_count = 0

    if is_camera:
        selected, camera_merged_detection_count = (
            consolidate_camera_signature(img, selected)
        )
        best_min_confidence = None
        best_dropped_count = max(
            0,
            post_nms_count - camera_merged_detection_count,
        )
    else:
        selected, best_min_confidence, best_dropped_count = (
            filter_best_confidence_detections(
                selected,
                debug_context=debug_context,
            )
        )

    debug_image_path = save_detection_debug_image(
        img,
        all_detections,
        selected,
        debug_context=debug_context,
    )
    camera_original_debug_image_path = save_camera_original_debug_image(
        img,
        all_detections,
        selected,
        debug_context=debug_context,
    )

    logger.info(
        "Detecciones finales context=%s pre_nms=%s post_nms=%s "
        "post_best=%s best_dropped=%s best_min_confidence=%s "
        "best_ratio=%.2f min_keep_confidence=%.2f "
        "nms_iou=%.2f nms_containment=%.2f",
        context_label(debug_context),
        len(all_detections),
        post_nms_count,
        len(selected),
        best_dropped_count,
        (
            round(best_min_confidence, 4)
            if best_min_confidence is not None
            else None
        ),
        SIGNATURE_BEST_CONFIDENCE_RATIO,
        SIGNATURE_MIN_KEEP_CONFIDENCE,
        SIGNATURE_DEDUP_IOU_THRESHOLD,
        SIGNATURE_DEDUP_CONTAINMENT_THRESHOLD,
    )

    detection_debug = {
        "image_width": int(w),
        "image_height": int(h),
        "variants": variants_debug,
        "pre_nms_count": len(all_detections),
        "post_nms_count": post_nms_count,
        "post_best_confidence_count": len(selected),
        "best_confidence_dropped_count": best_dropped_count,
        "best_confidence_ratio": SIGNATURE_BEST_CONFIDENCE_RATIO,
        "min_keep_confidence": SIGNATURE_MIN_KEEP_CONFIDENCE,
        "best_confidence_min_threshold": (
            round(best_min_confidence, 4)
            if best_min_confidence is not None
            else None
        ),
        "nms_iou_threshold": SIGNATURE_DEDUP_IOU_THRESHOLD,
        "nms_containment_threshold": SIGNATURE_DEDUP_CONTAINMENT_THRESHOLD,
        "debug_image_path": debug_image_path,
        "camera_original_debug_image_path": (
            camera_original_debug_image_path
        ),
        "camera_merged_detection_count": (
            camera_merged_detection_count
        ),
        "final_boxes": [
            summarize_final_detection(det)
            for det in selected
        ],
    }

    if return_debug:
        return selected, detection_debug

    return selected
