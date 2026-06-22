import os
import cv2
import uuid
import numpy as np

from datetime import datetime

from config import (
    MATCH_THRESHOLD,
    SEARCH_SCALES,
    SEARCH_SHIFT_X,
    SEARCH_SHIFT_Y,
    SEARCH_ROTATIONS,
    DEBUG_SAVE_FILES,
    DEBUG_DIR,
)

from compare_preprocess import (
    preprocess_signature,
    normalize_foreground,
)


def compute_iou(mask1, mask2):
    m1 = mask1 > 0
    m2 = mask2 > 0

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_dice(mask1, mask2):
    m1 = mask1 > 0
    m2 = mask2 > 0

    intersection = np.logical_and(m1, m2).sum()
    total = m1.sum() + m2.sum()

    if total == 0:
        return 0.0

    return float((2.0 * intersection) / total)


def compute_score(dice, iou):
    return (0.93 * dice) + (0.07 * iou)


def transform_mask(mask, scale, dx, dy, angle):
    h, w = mask.shape[:2]

    center = (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale,
    )

    matrix[0, 2] += dx
    matrix[1, 2] += dy

    transformed = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    _, transformed = cv2.threshold(
        transformed,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    return transformed
def build_overlay(upload_mask, reference_mask):
    h, w = upload_mask.shape[:2]

    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    ref = reference_mask > 0
    upl = upload_mask > 0
    both = np.logical_and(ref, upl)

    overlay[ref] = (0, 0, 255)
    overlay[upl] = (0, 255, 0)
    overlay[both] = (0, 255, 255)

    return overlay


def find_best_alignment(upload_aligned, reference_aligned):
    best = {
        "score": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "scale": 1.0,
        "dx": 0,
        "dy": 0,
        "angle": 0,
        "upload_best": upload_aligned.copy(),
        "reference_best": reference_aligned.copy(),
    }

    for angle in SEARCH_ROTATIONS:
        for scale in SEARCH_SCALES:
            for dx in SEARCH_SHIFT_X:
                for dy in SEARCH_SHIFT_Y:

                    moved = transform_mask(
                        upload_aligned,
                        scale,
                        dx,
                        dy,
                        angle,
                    )

                    dice = compute_dice(
                        moved,
                        reference_aligned,
                    )

                    iou = compute_iou(
                        moved,
                        reference_aligned,
                    )

                    score = compute_score(
                        dice,
                        iou,
                    )

                    if score > best["score"]:
                        best = {
                            "score": float(score),
                            "dice": float(dice),
                            "iou": float(iou),
                            "scale": float(scale),
                            "dx": int(dx),
                            "dy": int(dy),
                            "angle": int(angle),
                            "upload_best": moved.copy(),
                            "reference_best": reference_aligned.copy(),
                        }

    return best


def save_debug_bundle(
    camera_signature,
    document_signature,
    camera_pre,
    document_pre,
    best,
    debug_context,
):
    if not DEBUG_SAVE_FILES:
        return None

    codigo_cliente = debug_context.get("codigo_cliente", "sin_cliente")
    page = debug_context.get("page", "sin_pagina")
    sig_idx = debug_context.get("signature_index", "sin_firma")

    debug_id = (
        f"{codigo_cliente}_p{page}_f{sig_idx}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )

    folder = os.path.join(DEBUG_DIR, debug_id)

    os.makedirs(folder, exist_ok=True)

    paths = {
        "camera_original": os.path.join(folder, "01_camera_original.png"),
        "document_original": os.path.join(folder, "02_document_original.png"),
        "camera_binary": os.path.join(folder, "03_camera_binary.png"),
        "document_binary": os.path.join(folder, "04_document_binary.png"),
        "camera_crop": os.path.join(folder, "05_camera_crop.png"),
        "document_crop": os.path.join(folder, "06_document_crop.png"),
        "camera_aligned": os.path.join(folder, "07_camera_aligned.png"),
        "document_aligned": os.path.join(folder, "08_document_aligned.png"),
        "camera_best": os.path.join(folder, "09_camera_best.png"),
        "document_best": os.path.join(folder, "10_document_best.png"),
        "overlay_best": os.path.join(folder, "11_overlay_best.png"),
    }

    overlay = build_overlay(
        best["upload_best"],
        best["reference_best"],
    )

    cv2.imwrite(paths["camera_original"], camera_signature)
    cv2.imwrite(paths["document_original"], document_signature)

    cv2.imwrite(paths["camera_binary"], camera_pre["binary"])
    cv2.imwrite(paths["document_binary"], document_pre["binary"])

    cv2.imwrite(paths["camera_crop"], camera_pre["crop"])
    cv2.imwrite(paths["document_crop"], document_pre["crop"])

    cv2.imwrite(paths["camera_aligned"], camera_pre["aligned"])
    cv2.imwrite(paths["document_aligned"], document_pre["aligned"])

    cv2.imwrite(paths["camera_best"], best["upload_best"])
    cv2.imwrite(paths["document_best"], best["reference_best"])

    cv2.imwrite(paths["overlay_best"], overlay)

    return {
        "debug_id": debug_id,
        "debug_folder": folder,
        "debug_paths": paths,
    }


def compare_signatures(
    camera_signature,
    document_signature,
    debug_context=None,
):
    debug_context = debug_context or {}

    camera_pre = preprocess_signature(
        camera_signature
    )

    document_pre = preprocess_signature(
        document_signature
    )

    camera_pixels = int(
        np.count_nonzero(
            camera_pre["aligned"]
        )
    )

    document_pixels = int(
        np.count_nonzero(
            document_pre["aligned"]
        )
    )

    if camera_pixels < 20:
        raise ValueError(
            "Firma camara sin suficiente trazo"
        )

    if document_pixels < 20:
        raise ValueError(
            "Firma documento sin suficiente trazo"
        )

    best = find_best_alignment(
        camera_pre["aligned"],
        document_pre["aligned"],
    )

    score = float(best["score"])

    debug_info = save_debug_bundle(
        camera_signature=camera_signature,
        document_signature=document_signature,
        camera_pre=camera_pre,
        document_pre=document_pre,
        best=best,
        debug_context=debug_context,
    )

    result = {
        "match": score >= MATCH_THRESHOLD,
        "score": round(score, 4),
        "dice": round(best["dice"], 4),
        "iou": round(best["iou"], 4),
        "threshold": MATCH_THRESHOLD,
        "best_alignment": {
            "scale": best["scale"],
            "dx": best["dx"],
            "dy": best["dy"],
            "angle": best["angle"],
        },
        "stroke_pixels": {
            "camera": camera_pixels,
            "document": document_pixels,
        },
    }

    if debug_info:
        result["debug_compare"] = debug_info

    return result