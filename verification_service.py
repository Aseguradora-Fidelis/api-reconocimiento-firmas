from compare_service import compare_signatures
from detection_service import detect_signatures
from pdf_service import extract_signature_candidates_from_pdf
from s3_oracle_service import (
    get_client_documents,
    get_pdf_from_s3,
    get_pdf_view_url,
)
from watermark import build_watermarked_base64, build_watermarked_signature_base64


def flatten_pdf_results(pdf_results):
    candidates = []

    for page_result in pdf_results:
        page_number = page_result["page"]

        for signature in page_result["signatures"]:
            candidates.append({
                "page": page_number,
                "signature": signature,
            })

    return sorted(
        candidates,
        key=lambda item: item["signature"].get(
            "candidate_rank",
            999999,
        ),
    )


def verify_signature(codigo_cliente: int, camera_signature):
    camera_detections = detect_signatures(camera_signature)

    if not camera_detections:
        return {
            "ok": False,
            "match": False,
            "codigo_cliente": str(codigo_cliente),
            "message": "No se detecto firma en la imagen de camara",
            "images": {
                "camera_image_base64": build_watermarked_base64(
                    camera_signature
                ),
                "camera_signature_base64": None,
                "compared_document_signatures": [],
            },
            "debug": {
                "camera_signatures_detected": 0,
                "documents_found": 0,
                "pdfs_read": 0,
                "pages_with_signatures": 0,
                "signatures_compared": 0,
            },
        }

    camera_signature_crop = camera_detections[0]["crop"]

    documents = get_client_documents(codigo_cliente)

    if not documents:
        return {
            "ok": False,
            "match": False,
            "codigo_cliente": str(codigo_cliente),
            "message": "Cliente sin documentos",
            "images": {
                "camera_signature_base64": build_watermarked_signature_base64(
                    camera_signature_crop
                ),
                "compared_document_signatures": [],
            },
            "debug": {
                "camera_signatures_detected": len(camera_detections),
                "documents_found": 0,
                "pdfs_read": 0,
                "pages_with_signatures": 0,
                "signatures_compared": 0,
            },
        }

    pdfs_read = 0
    pages_with_signatures = 0
    signatures_compared = 0
    errors = []
    compared_signatures = []
    file_view_urls = {}
    pdf_extraction_debug = []

    client_name = None
    for document in documents:
        if document.get("nombre_cliente"):
            client_name = document["nombre_cliente"]
            break

    def resolve_file_view_url(s3_key):
        if s3_key not in file_view_urls:
            file_view_urls[s3_key] = get_pdf_view_url(s3_key)

        return file_view_urls[s3_key]

    best_attempt = {
        "score": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "threshold": None,
        "archivo": None,
        "s3_key": None,
        "file_view_url": None,
        "page": None,
        "signature_index": None,
    }

    for document in documents:
        archivo = document["archivo"]
        s3_key = document["s3_key"]

        pdf_buffer = get_pdf_from_s3(s3_key)

        if pdf_buffer is None:
            errors.append(f"No se pudo leer PDF S3: {s3_key}")
            continue

        pdfs_read += 1

        try:
            extracted = extract_signature_candidates_from_pdf(
                pdf_buffer=pdf_buffer,
                stop_at_first_page=False,
                client_name=client_name,
            )
        except Exception as e:
            errors.append(f"Error extrayendo firmas de {archivo}: {e}")
            continue

        pdf_results = extracted["pages"]
        extraction_debug = extracted["debug"]
        extraction_debug["archivo"] = archivo
        extraction_debug["s3_key"] = s3_key
        pdf_extraction_debug.append(extraction_debug)

        candidate_pages = {
            page_result["page"]
            for page_result in pdf_results
            if page_result["signatures"]
        }
        pages_with_signatures += len(candidate_pages)

        candidates = flatten_pdf_results(pdf_results)

        for candidate in candidates:
            page_number = candidate["page"]
            signature = candidate["signature"]
            signature_index = signature["signature_index"]
            document_signature = signature["image"]
            name_match = signature.get("name_match") or {}

            try:
                compare_result = compare_signatures(
                    camera_signature_crop,
                    document_signature,
                    debug_context={
                        "codigo_cliente": codigo_cliente,
                        "archivo": archivo,
                        "s3_key": s3_key,
                        "page": page_number,
                        "signature_index": signature_index,
                        "candidate_rank": signature.get("candidate_rank"),
                        "name_match_status": name_match.get("status"),
                    },
                )
            except Exception as e:
                errors.append(
                    f"Error comparando {archivo} pagina {page_number} "
                    f"firma {signature_index}: {e}"
                )
                continue

            signatures_compared += 1

            score = float(compare_result["score"])
            file_view_url = resolve_file_view_url(s3_key)

            compared_signatures.append({
                "archivo": archivo,
                "s3_key": s3_key,
                "file_view_url": file_view_url,
                "page": page_number,
                "signature_index": signature_index,
                "candidate_rank": signature.get("candidate_rank"),
                "name_text": signature.get("name_text"),
                "name_text_source": signature.get("name_text_source"),
                "name_match": name_match,
                "score": compare_result["score"],
                "dice": compare_result["dice"],
                "iou": compare_result["iou"],
                "threshold": compare_result["threshold"],
                "image": document_signature.copy(),
            })

            if score > best_attempt["score"]:
                best_attempt = {
                    "score": compare_result["score"],
                    "dice": compare_result["dice"],
                    "iou": compare_result["iou"],
                    "threshold": compare_result["threshold"],
                    "archivo": archivo,
                    "s3_key": s3_key,
                    "file_view_url": file_view_url,
                    "page": page_number,
                    "signature_index": signature_index,
                    "candidate_rank": signature.get("candidate_rank"),
                    "name_text": signature.get("name_text"),
                    "name_text_source": signature.get("name_text_source"),
                    "name_match": name_match,
                    "debug_compare": compare_result.get("debug_compare"),
                }

            if compare_result["match"]:
                return {
                    "ok": True,
                    "match": True,
                    "codigo_cliente": str(codigo_cliente),
                    "score": compare_result["score"],
                    "dice": compare_result["dice"],
                    "iou": compare_result["iou"],
                    "threshold": compare_result["threshold"],
                    "document_match": {
                        "archivo": archivo,
                        "s3_key": s3_key,
                        "file_view_url": file_view_url,
                        "page": page_number,
                        "signature_index": signature_index,
                        "candidate_rank": signature.get("candidate_rank"),
                        "name_text": signature.get("name_text"),
                        "name_text_source": signature.get("name_text_source"),
                        "name_match": name_match,
                    },
                    "images": {
                        "camera_signature_base64": build_watermarked_signature_base64(
                            camera_signature_crop
                        ),
                        "matched_document_signature_base64": build_watermarked_signature_base64(
                            document_signature
                        ),
                    },
                    "debug": {
                        "camera_signatures_detected": len(camera_detections),
                        "client_name": client_name,
                        "documents_found": len(documents),
                        "pdfs_read": pdfs_read,
                        "pages_with_signatures": pages_with_signatures,
                        "signatures_compared": signatures_compared,
                        "early_stop": True,
                        "pdf_extraction": pdf_extraction_debug,
                        "errors": errors,
                        "debug_compare": compare_result.get("debug_compare"),
                    },
                }

    return {
        "ok": True,
        "match": False,
        "codigo_cliente": str(codigo_cliente),
        "message": "No se encontro coincidencia",
        "best_attempt": best_attempt,
        "images": {
            "camera_signature_base64": build_watermarked_signature_base64(
                camera_signature_crop
            ),
            "compared_document_signatures": [
                {
                    "archivo": item["archivo"],
                    "s3_key": item["s3_key"],
                    "file_view_url": item["file_view_url"],
                    "page": item["page"],
                    "signature_index": item["signature_index"],
                    "candidate_rank": item["candidate_rank"],
                    "name_text": item["name_text"],
                    "name_text_source": item["name_text_source"],
                    "name_match": item["name_match"],
                    "score": item["score"],
                    "dice": item["dice"],
                    "iou": item["iou"],
                    "threshold": item["threshold"],
                    "document_signature_base64": build_watermarked_signature_base64(
                        item["image"]
                    ),
                }
                for item in compared_signatures
            ],
        },
        "debug": {
            "camera_signatures_detected": len(camera_detections),
            "client_name": client_name,
            "documents_found": len(documents),
            "pdfs_read": pdfs_read,
            "pages_with_signatures": pages_with_signatures,
            "signatures_compared": signatures_compared,
            "early_stop": False,
            "pdf_extraction": pdf_extraction_debug,
            "errors": errors,
        },
    }
