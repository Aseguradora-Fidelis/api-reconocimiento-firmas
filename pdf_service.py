from io import BytesIO

import cv2
import fitz
import numpy as np

from config import PDF_DPI
from detection_service import detect_signatures


def pdf_to_images(pdf_buffer: BytesIO, dpi: int = PDF_DPI):
    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc[page_index]

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

        if pix.n == 3:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR,
            )

        images.append({
            "page": page_index + 1,
            "image": img,
        })

    doc.close()
    return images


def extract_signatures_from_pdf(
    pdf_buffer: BytesIO,
    stop_at_first_page: bool = True,
):
    pages = pdf_to_images(pdf_buffer)

    results = []

    for item in pages:
        page_number = item["page"]
        page_img = item["image"]

        detections = detect_signatures(page_img)

        if not detections:
            continue

        signatures = []

        for idx, detection in enumerate(
            detections,
            start=1,
        ):
            crop = detection["crop"]

            if crop is None or crop.size == 0:
                continue

            signatures.append({
                "signature_index": idx,
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "image": crop.copy(),
            })

        if signatures:
            results.append({
                "page": page_number,
                "signatures": signatures,
            })

            if stop_at_first_page:
                break

    return results