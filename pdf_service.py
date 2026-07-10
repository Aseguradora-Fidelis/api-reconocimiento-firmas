from io import BytesIO
from difflib import SequenceMatcher
import logging
import re
import unicodedata

import cv2
import fitz
import numpy as np

from config import (
    PDF_DPI,
    PDF_MAX_PAGES_TO_SCAN,
    PDF_MAX_SIGNATURES_TO_COMPARE,
    PDF_MAX_SIGNATURES_PER_PAGE,
    PDF_NAME_MATCH_THRESHOLD,
    PDF_OCR_LANG,
)
from detection_service import detect_signatures
from image_utils import clamp_bbox, crop_from_bbox

logger = logging.getLogger(__name__)

try:
    import pytesseract
except ImportError:
    pytesseract = None


NAME_STOPWORDS = {
    "firma",
    "firmante",
    "nombre",
    "cliente",
    "contacto",
    "del",
    "las",
    "los",
    "representante",
    "legal",
    "autorizado",
    "asegurado",
    "solicitante",
    "propietario",
    "deudor",
    "codeudor",
    "dpi",
    "nit",
    "fecha",
    "lugar",
    "documento",
    "identificacion",
}


def normalize_text(value):
    if not value:
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        ch
        for ch in value
        if not unicodedata.combining(ch)
    )
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def name_tokens(value):
    normalized = normalize_text(value)

    return [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in NAME_STOPWORDS
    ]


def match_client_name(text, client_name):
    client_normalized = normalize_text(client_name)
    text_normalized = normalize_text(text)

    if not client_normalized:
        return {
            "status": "not_available",
            "match": False,
            "score": 0.0,
            "matched_tokens": [],
        }

    if not text_normalized:
        return {
            "status": "unknown",
            "match": False,
            "score": 0.0,
            "matched_tokens": [],
        }

    client_tokens = name_tokens(client_name)
    text_tokens = set(name_tokens(text))

    if not text_tokens:
        return {
            "status": "unknown",
            "match": False,
            "score": 0.0,
            "matched_tokens": [],
        }

    if client_normalized in text_normalized:
        return {
            "status": "match",
            "match": True,
            "score": 1.0,
            "matched_tokens": client_tokens,
        }

    matched_tokens = [
        token
        for token in client_tokens
        if token in text_tokens
    ]

    token_score = 0.0
    if client_tokens:
        token_score = len(matched_tokens) / len(client_tokens)

    ratio_score = SequenceMatcher(
        None,
        client_normalized,
        text_normalized,
    ).ratio()

    score = max(token_score, ratio_score)

    minimum_matches = 1
    if len(client_tokens) > 1:
        minimum_matches = 2

    is_match = (
        score >= PDF_NAME_MATCH_THRESHOLD
        and len(matched_tokens) >= minimum_matches
    )

    return {
        "status": "match" if is_match else "mismatch",
        "match": bool(is_match),
        "score": round(float(score), 4),
        "matched_tokens": matched_tokens,
    }


def render_page_to_image(page, dpi: int = PDF_DPI):
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8,
    )

    img = img.reshape(
        pix.height,
        pix.width,
        pix.n,
    )

    if pix.n == 1:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR,
        )
    elif pix.n == 3:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR,
        )
    elif pix.n == 4:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2BGR,
        )

    return img, zoom


def expand_signature_context_bbox(bbox, page_shape):
    page_h, page_w = page_shape[:2]

    x1 = bbox["x1"]
    y1 = bbox["y1"]
    x2 = bbox["x2"]
    y2 = bbox["y2"]

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    x_margin = max(20, int(width * 0.35))
    top_margin = max(10, int(height * 0.15))
    bottom_margin = max(60, int(height * 1.60))

    cx1, cy1, cx2, cy2 = clamp_bbox(
        x1 - x_margin,
        y1 - top_margin,
        x2 + x_margin,
        y2 + bottom_margin,
        page_w,
        page_h,
    )

    return {
        "x1": int(cx1),
        "y1": int(cy1),
        "x2": int(cx2),
        "y2": int(cy2),
    }


def bbox_to_pdf_rect(bbox, zoom):
    return fitz.Rect(
        bbox["x1"] / zoom,
        bbox["y1"] / zoom,
        bbox["x2"] / zoom,
        bbox["y2"] / zoom,
    )


def clean_extracted_text(text):
    if not text:
        return ""

    return " ".join(str(text).split())


def extract_native_text_from_bbox(page, bbox, zoom):
    rect = bbox_to_pdf_rect(bbox, zoom)

    try:
        text = page.get_text(
            "text",
            clip=rect,
        )
    except Exception:
        return ""

    return clean_extracted_text(text)


def prepare_ocr_image(img):
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY,
    )

    _, w = gray.shape[:2]

    scale = 1.0
    if w < 700:
        scale = 700 / max(1, w)

    if scale > 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )


def extract_ocr_text(img):
    if pytesseract is None:
        return "", "pytesseract_not_available"

    try:
        ocr_img = prepare_ocr_image(img)
        text = pytesseract.image_to_string(
            ocr_img,
            lang=PDF_OCR_LANG,
            config="--psm 6",
        )

        return clean_extracted_text(text), None

    except Exception as e:
        return "", str(e)


def candidate_sort_key(candidate):
    status_order = {
        "match": 0,
        "unknown": 1,
        "not_available": 1,
        "mismatch": 2,
    }

    name_match = candidate.get("name_match") or {}
    status = name_match.get("status", "unknown")

    return (
        status_order.get(status, 1),
        -float(name_match.get("score") or 0.0),
        candidate["page"],
        candidate["signature_index"],
        -float(candidate.get("confidence") or 0.0),
    )


def group_candidates_by_page(candidates):
    pages = []
    page_map = {}

    for candidate in candidates:
        page_number = candidate["page"]

        if page_number not in page_map:
            page_result = {
                "page": page_number,
                "signatures": [],
            }
            page_map[page_number] = page_result
            pages.append(page_result)

        page_map[page_number]["signatures"].append({
            "signature_index": candidate["signature_index"],
            "candidate_rank": candidate["candidate_rank"],
            "bbox": candidate["bbox"],
            "context_bbox": candidate["context_bbox"],
            "confidence": candidate["confidence"],
            "image": candidate["image"],
            "context_image": candidate["context_image"],
            "name_text": candidate["name_text"],
            "name_text_source": candidate["name_text_source"],
            "name_match": candidate["name_match"],
            "ocr_error": candidate["ocr_error"],
        })

    return pages


def pdf_to_images(pdf_buffer: BytesIO, dpi: int = PDF_DPI):
    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    images = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        img, _ = render_page_to_image(page, dpi=dpi)

        images.append({
            "page": page_index + 1,
            "image": img,
        })

    doc.close()
    return images


def extract_signature_candidates_from_pdf(
    pdf_buffer: BytesIO,
    stop_at_first_page: bool = False,
    client_name: str | None = None,
    max_pages_to_scan: int = PDF_MAX_PAGES_TO_SCAN,
    max_signatures_to_compare: int = PDF_MAX_SIGNATURES_TO_COMPARE,
    max_signatures_per_page: int = PDF_MAX_SIGNATURES_PER_PAGE,
    debug_context: dict | None = None,
):
    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    total_pages = len(doc)

    if max_pages_to_scan is None or max_pages_to_scan <= 0:
        pages_to_scan = total_pages
    else:
        pages_to_scan = min(total_pages, max_pages_to_scan)

    debug = {
        "total_pages": total_pages,
        "pages_scanned": 0,
        "max_pages_to_scan": max_pages_to_scan,
        "max_signatures_to_compare": max_signatures_to_compare,
        "max_signatures_per_page": max_signatures_per_page,
        "pdf_dpi": PDF_DPI,
        "signatures_detected": 0,
        "signatures_selected_before_ranking": 0,
        "signatures_returned_for_comparison": 0,
        "detection_pages": [],
        "name_matches": 0,
        "name_mismatches": 0,
        "name_unknown": 0,
        "ocr_available": pytesseract is not None,
        "ocr_attempts": 0,
        "ocr_skipped_reason": (
            None
            if pytesseract is not None
            else "pytesseract_not_available"
        ),
        "ocr_errors": [],
    }

    selected_candidates = []

    try:
        for page_index in range(pages_to_scan):
            page = doc[page_index]
            page_number = page_index + 1
            page_img, zoom = render_page_to_image(
                page,
                dpi=PDF_DPI,
            )

            debug["pages_scanned"] += 1

            page_debug_context = {
                **(debug_context or {}),
                "source": "pdf_page",
                "page": page_number,
            }

            detections, detection_debug = detect_signatures(
                page_img,
                debug_context=page_debug_context,
                return_debug=True,
            )
            debug["detection_pages"].append({
                "page": page_number,
                "render_dpi": PDF_DPI,
                "render_width": int(page_img.shape[1]),
                "render_height": int(page_img.shape[0]),
                **detection_debug,
            })
            debug["signatures_detected"] += len(detections)

            logger.info(
                "Pagina PDF analizada page=%s dpi=%s size=%sx%s "
                "pre_nms=%s post_nms=%s post_best=%s",
                page_number,
                PDF_DPI,
                page_img.shape[1],
                page_img.shape[0],
                detection_debug["pre_nms_count"],
                detection_debug["post_nms_count"],
                detection_debug["post_best_confidence_count"],
            )

            if not detections:
                continue

            page_candidates = []

            for idx, detection in enumerate(
                detections,
                start=1,
            ):
                crop = detection["crop"]

                if crop is None or crop.size == 0:
                    continue

                bbox = detection["bbox"]
                context_bbox = expand_signature_context_bbox(
                    bbox,
                    page_img.shape,
                )

                context_crop = crop_from_bbox(
                    page_img,
                    context_bbox,
                )

                name_text = extract_native_text_from_bbox(
                    page,
                    context_bbox,
                    zoom,
                )
                name_text_source = "native" if name_text else None
                ocr_error = None

                if (
                    not name_text
                    and context_crop is not None
                    and pytesseract is not None
                ):
                    debug["ocr_attempts"] += 1
                    name_text, ocr_error = extract_ocr_text(
                        context_crop
                    )

                    if name_text:
                        name_text_source = "ocr"
                    elif ocr_error:
                        debug["ocr_errors"].append({
                            "page": page_number,
                            "signature_index": idx,
                            "error": ocr_error,
                        })
                elif not name_text and context_crop is not None:
                    ocr_error = debug["ocr_skipped_reason"]

                name_match = match_client_name(
                    name_text,
                    client_name,
                )

                status = name_match["status"]
                if status == "match":
                    debug["name_matches"] += 1
                elif status == "mismatch":
                    debug["name_mismatches"] += 1
                else:
                    debug["name_unknown"] += 1

                page_candidates.append({
                    "page": page_number,
                    "signature_index": idx,
                    "bbox": bbox,
                    "context_bbox": context_bbox,
                    "confidence": detection["confidence"],
                    "image": crop.copy(),
                    "context_image": (
                        context_crop.copy()
                        if context_crop is not None
                        else None
                    ),
                    "name_text": name_text,
                    "name_text_source": name_text_source,
                    "name_match": name_match,
                    "ocr_error": ocr_error,
                })

            page_candidates = sorted(
                page_candidates,
                key=candidate_sort_key,
            )

            if (
                max_signatures_per_page
                and max_signatures_per_page > 0
                and len(page_candidates) > max_signatures_per_page
            ):
                logger.warning(
                    "Recortando firmas por pagina page=%s total=%s limit=%s",
                    page_number,
                    len(page_candidates),
                    max_signatures_per_page,
                )
                page_candidates = page_candidates[:max_signatures_per_page]

            selected_candidates.extend(page_candidates)
            debug["signatures_selected_before_ranking"] = len(
                selected_candidates
            )

            if stop_at_first_page and page_candidates:
                break

        selected_candidates = sorted(
            selected_candidates,
            key=candidate_sort_key,
        )

        if (
            max_signatures_to_compare
            and max_signatures_to_compare > 0
            and len(selected_candidates) > max_signatures_to_compare
        ):
            logger.warning(
                "Recortando firmas totales total=%s limit=%s",
                len(selected_candidates),
                max_signatures_to_compare,
            )
            selected_candidates = selected_candidates[
                :max_signatures_to_compare
            ]

        for rank, candidate in enumerate(
            selected_candidates,
            start=1,
        ):
            candidate["candidate_rank"] = rank

        debug["signatures_returned_for_comparison"] = len(
            selected_candidates
        )

        return {
            "pages": group_candidates_by_page(selected_candidates),
            "debug": debug,
        }

    finally:
        doc.close()


def extract_signatures_from_pdf(
    pdf_buffer: BytesIO,
    stop_at_first_page: bool = False,
):
    extracted = extract_signature_candidates_from_pdf(
        pdf_buffer=pdf_buffer,
        stop_at_first_page=stop_at_first_page,
        max_pages_to_scan=0,
        max_signatures_to_compare=0,
        max_signatures_per_page=0,
    )

    return extracted["pages"]
