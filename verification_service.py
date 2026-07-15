from audit_service import persist_verification_audit
from compare_service import compare_signatures
from detection_service import detect_signatures
from pdf_service import (
    extract_ocr_text,
    extract_signature_candidates_from_pdf,
    name_tokens,
    normalize_text,
)
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


def summarize_attempt(attempt):
    return {
        "score": attempt.get("score"),
        "dice": attempt.get("dice"),
        "iou": attempt.get("iou"),
        "threshold": attempt.get("threshold"),
        "archivo": attempt.get("archivo"),
        "s3_key": attempt.get("s3_key"),
        "file_view_url": attempt.get("file_view_url"),
        "page": attempt.get("page"),
        "signature_index": attempt.get("signature_index"),
        "candidate_rank": attempt.get("candidate_rank"),
        "name_text": attempt.get("name_text"),
        "name_text_source": attempt.get("name_text_source"),
        "name_match": attempt.get("name_match"),
        "visual_match": attempt.get("visual_match"),
        "name_eligible": attempt.get("name_eligible"),
        "match": attempt.get("match"),
    }


def resolve_ocr_client_name(documents, camera_signature=None):
    references = []
    seen_names = set()
    representative_code = None

    for document in documents:
        for field, source in (
            ("nombre_representante_legal", "representante_legal"),
            ("nombre_cliente", "nombre_cliente"),
        ):
            name = document.get(field)
            normalized = normalize_text(name)

            # Valores placeholder como "." no son nombres utilizables.
            if not normalized or not name_tokens(name):
                continue

            if normalized in seen_names:
                continue

            seen_names.add(normalized)
            references.append({
                "name": name,
                "source": source,
            })

            if source == "representante_legal":
                representative_code = document.get(
                    "codigo_representante_legal"
                )

    if references:
        sources = {item["source"] for item in references}
        source = (
            "cliente_y_representante_legal"
            if len(sources) > 1
            else references[0]["source"]
        )

        return {
            "name": references[0]["name"],
            "names": [item["name"] for item in references],
            "name_references": references,
            "source": source,
            "codigo_representante_legal": representative_code,
            "camera_ocr_text": None,
            "camera_ocr_error": None,
        }

    camera_ocr_text = ""
    camera_ocr_error = None

    if camera_signature is not None:
        camera_ocr_text, camera_ocr_error = extract_ocr_text(
            camera_signature
        )
        camera_tokens = name_tokens(camera_ocr_text)

        if (
            len(camera_tokens) >= 3
            and sum(len(token) for token in camera_tokens) >= 12
        ):
            return {
                "name": camera_ocr_text,
                "names": [camera_ocr_text],
                "name_references": [{
                    "name": camera_ocr_text,
                    "source": "camera_ocr_fallback",
                }],
                "source": "camera_ocr_fallback",
                "codigo_representante_legal": None,
                "camera_ocr_text": camera_ocr_text,
                "camera_ocr_error": camera_ocr_error,
            }

    return {
        "name": None,
        "names": [],
        "name_references": [],
        "source": None,
        "codigo_representante_legal": None,
        "camera_ocr_text": camera_ocr_text or None,
        "camera_ocr_error": camera_ocr_error,
    }


def name_status_priority(name_match, name_required):
    if not name_required:
        return 0

    status = (name_match or {}).get("status")

    return {
        "match": 2,
        "unknown": 1,
        "not_available": 1,
        "mismatch": 0,
    }.get(status, 1)


def candidate_selection_key(name_match, score, name_required):
    return (
        name_status_priority(name_match, name_required),
        float(score),
    )


def name_is_eligible(name_match, name_required):
    if not name_required:
        return True

    return (name_match or {}).get("status") == "match"


def first_document_value(documents, key):
    for document in documents:
        value = document.get(key)

        if value:
            return value

    return None


def build_candidate_audit(page_number, signature):
    return {
        "page": page_number,
        "signature_index": signature.get("signature_index"),
        "candidate_rank": signature.get("candidate_rank"),
        "confidence": signature.get("confidence"),
        "bbox": signature.get("bbox"),
        "context_bbox": signature.get("context_bbox"),
        "candidate_image": signature.get("image"),
        "context_image": signature.get("context_image"),
        "name_text": signature.get("name_text"),
        "name_text_source": signature.get("name_text_source"),
        "name_match": signature.get("name_match"),
        "compared": False,
        "compare_result": None,
        "is_best": False,
    }


def build_audit_payload(
    codigo_cliente,
    documents,
    ocr_name_info,
    status,
    match_automatico,
    best_attempt,
    errors,
    audit_documents,
    condicion_entrega_id=None,
    fianza=None,
    camera_signature_image=None,
):
    return {
        "codigo_cliente": codigo_cliente,
        "condicion_entrega_id": condicion_entrega_id,
        "fianza": fianza,
        "codigo_representante_legal": ocr_name_info.get(
            "codigo_representante_legal"
        ),
        "nombre_cliente": first_document_value(
            documents,
            "nombre_cliente",
        ),
        "nombre_representante_legal": first_document_value(
            documents,
            "nombre_representante_legal",
        ),
        "status": status,
        "match_automatico": match_automatico,
        "best_score": best_attempt.get("score"),
        "request": {
            "codigo_cliente": codigo_cliente,
            "condicion_entrega_id": condicion_entrega_id,
            "fianza": fianza,
        },
        "error_message": "; ".join(errors) if errors else None,
        "camera_signature_image": camera_signature_image,
        "documents": audit_documents,
    }


def attach_audit(
    result,
    audit_payload,
    background_tasks=None,
):
    audit_result = persist_verification_audit(
        audit_payload,
        result,
        background_tasks=background_tasks,
    )

    result.setdefault("debug", {})["audit"] = audit_result

    return result


def verify_signature(
    codigo_cliente: int,
    camera_signature,
    condicion_entrega_id: int | None = None,
    fianza: int | None = None,
    background_tasks=None,
):
    camera_detections, camera_detection_debug = detect_signatures(
        camera_signature,
        debug_context={
            "source": "camera",
        },
        return_debug=True,
    )

    if not camera_detections:
        result = {
            "ok": False,
            "match": False,
            "codigo_cliente": str(codigo_cliente),
            "condicion_entrega_id": condicion_entrega_id,
            "fianza": fianza,
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
                "camera_detection": camera_detection_debug,
                "documents_found": 0,
                "pdfs_read": 0,
                "pages_with_signatures": 0,
                "signatures_compared": 0,
            },
        }

        audit_payload = build_audit_payload(
            codigo_cliente=codigo_cliente,
            documents=[],
            ocr_name_info={
                "codigo_representante_legal": None,
            },
            status="no_match",
            match_automatico=False,
            best_attempt={},
            errors=["No se detecto firma en la imagen de camara"],
            audit_documents=[],
            condicion_entrega_id=condicion_entrega_id,
            fianza=fianza,
            camera_signature_image=None,
        )

        return attach_audit(
            result,
            audit_payload,
            background_tasks=background_tasks,
        )

    camera_signature_crop = camera_detections[0]["crop"]

    documents = get_client_documents(
        codigo_cliente,
        fianza=fianza,
    )

    if not documents:
        result = {
            "ok": False,
            "match": False,
            "codigo_cliente": str(codigo_cliente),
            "condicion_entrega_id": condicion_entrega_id,
            "fianza": fianza,
            "message": "Cliente sin documentos",
            "images": {
                "camera_signature_base64": build_watermarked_signature_base64(
                    camera_signature_crop
                ),
                "compared_document_signatures": [],
            },
            "debug": {
                "camera_signatures_detected": len(camera_detections),
                "camera_detection": camera_detection_debug,
                "documents_found": 0,
                "pdfs_read": 0,
                "pages_with_signatures": 0,
                "signatures_compared": 0,
            },
        }

        audit_payload = build_audit_payload(
            codigo_cliente=codigo_cliente,
            documents=[],
            ocr_name_info={
                "codigo_representante_legal": None,
            },
            status="no_match",
            match_automatico=False,
            best_attempt={},
            errors=["Cliente sin documentos"],
            audit_documents=[],
            condicion_entrega_id=condicion_entrega_id,
            fianza=fianza,
            camera_signature_image=camera_signature_crop,
        )

        return attach_audit(
            result,
            audit_payload,
            background_tasks=background_tasks,
        )

    pdfs_read = 0
    pages_with_signatures = 0
    signatures_compared = 0
    errors = []
    compared_signatures = []
    file_view_urls = {}
    pdf_extraction_debug = []
    audit_documents = []
    best_audit_candidate = None
    best_compared_signature = None
    best_selection_key = None
    has_match = False

    ocr_name_info = resolve_ocr_client_name(
        documents,
        camera_signature=camera_signature_crop,
    )
    client_name = ocr_name_info["name"]
    client_names = ocr_name_info["names"]
    name_required = bool(client_names)

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
        "candidate_rank": None,
        "name_text": None,
        "name_text_source": None,
        "name_match": None,
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
                client_name=client_names,
                debug_context={
                    "archivo": archivo,
                    "s3_key": s3_key,
                },
            )
        except Exception as e:
            errors.append(f"Error extrayendo firmas de {archivo}: {e}")
            continue

        pdf_results = extracted["pages"]
        extraction_debug = extracted["debug"]
        extraction_debug["archivo"] = archivo
        extraction_debug["s3_key"] = s3_key
        pdf_extraction_debug.append(extraction_debug)

        document_audit = {
            "archivo": archivo,
            "s3_key": s3_key,
            "debug": extraction_debug,
            "candidates": [],
        }
        audit_documents.append(document_audit)

        candidate_pages = {
            page_result["page"]
            for page_result in pdf_results
            if page_result["signatures"]
        }
        pages_with_signatures += len(candidate_pages)

        candidates = flatten_pdf_results(pdf_results)
        candidate_audits = {}

        for candidate in candidates:
            page_number = candidate["page"]
            signature = candidate["signature"]
            candidate_key = (
                page_number,
                signature["signature_index"],
            )
            candidate_audit = build_candidate_audit(
                page_number,
                signature,
            )
            document_audit["candidates"].append(candidate_audit)
            candidate_audits[candidate_key] = candidate_audit

        for candidate in candidates:
            page_number = candidate["page"]
            signature = candidate["signature"]
            signature_index = signature["signature_index"]
            document_signature = signature["image"]
            name_match = signature.get("name_match") or {}
            candidate_audit = candidate_audits[
                (
                    page_number,
                    signature_index,
                )
            ]

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
            visual_match = bool(compare_result["match"])
            name_eligible = name_is_eligible(
                name_match,
                name_required,
            )
            effective_match = visual_match and name_eligible
            candidate_audit["compared"] = True
            candidate_audit["compare_result"] = {
                "match": effective_match,
                "visual_match": visual_match,
                "name_eligible": name_eligible,
                "score": compare_result["score"],
                "dice": compare_result["dice"],
                "iou": compare_result["iou"],
                "threshold": compare_result["threshold"],
            }

            score = float(compare_result["score"])
            file_view_url = resolve_file_view_url(s3_key)

            compared_signature = {
                "archivo": archivo,
                "s3_key": s3_key,
                "file_view_url": file_view_url,
                "page": page_number,
                "signature_index": signature_index,
                "candidate_rank": signature.get("candidate_rank"),
                "name_text": signature.get("name_text"),
                "name_text_source": signature.get("name_text_source"),
                "name_match": name_match,
                "match": effective_match,
                "visual_match": visual_match,
                "name_eligible": name_eligible,
                "score": compare_result["score"],
                "dice": compare_result["dice"],
                "iou": compare_result["iou"],
                "threshold": compare_result["threshold"],
                "image": document_signature.copy(),
            }
            compared_signatures.append(compared_signature)
            selection_key = candidate_selection_key(
                name_match,
                score,
                name_required,
            )

            if (
                best_audit_candidate is None
                or selection_key > best_selection_key
            ):
                if best_audit_candidate is not None:
                    best_audit_candidate["is_best"] = False

                candidate_audit["is_best"] = True
                best_audit_candidate = candidate_audit
                best_selection_key = selection_key

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
                    "visual_match": visual_match,
                    "name_eligible": name_eligible,
                    "match": effective_match,
                    "debug_compare": compare_result.get("debug_compare"),
                }
                best_compared_signature = compared_signature

            if effective_match:
                has_match = True

    compared_document_signatures = [
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
            "match": item["match"],
            "visual_match": item["visual_match"],
            "name_eligible": item["name_eligible"],
            "score": item["score"],
            "dice": item["dice"],
            "iou": item["iou"],
            "threshold": item["threshold"],
            "document_signature_base64": build_watermarked_signature_base64(
                item["image"]
            ),
        }
        for item in compared_signatures
    ]

    if has_match and best_compared_signature:
        result = {
            "ok": True,
            "match": True,
            "codigo_cliente": str(codigo_cliente),
            "condicion_entrega_id": condicion_entrega_id,
            "fianza": fianza,
            "score": best_attempt["score"],
            "dice": best_attempt["dice"],
            "iou": best_attempt["iou"],
            "threshold": best_attempt["threshold"],
            "document_match": {
                "archivo": best_compared_signature["archivo"],
                "s3_key": best_compared_signature["s3_key"],
                "file_view_url": best_compared_signature["file_view_url"],
                "page": best_compared_signature["page"],
                "signature_index": best_compared_signature[
                    "signature_index"
                ],
                "candidate_rank": best_compared_signature[
                    "candidate_rank"
                ],
                "name_text": best_compared_signature["name_text"],
                "name_text_source": best_compared_signature[
                    "name_text_source"
                ],
                "name_match": best_compared_signature["name_match"],
            },
            "images": {
                "camera_signature_base64": build_watermarked_signature_base64(
                    camera_signature_crop
                ),
                "matched_document_signature_base64": build_watermarked_signature_base64(
                    best_compared_signature["image"]
                ),
                "compared_document_signatures": compared_document_signatures,
            },
            "debug": {
                "camera_signatures_detected": len(camera_detections),
                "camera_detection": camera_detection_debug,
                "client_name": client_name,
                "client_names": client_names,
                "client_name_references": ocr_name_info.get(
                    "name_references"
                ),
                "client_name_source": ocr_name_info["source"],
                "camera_name_ocr_text": ocr_name_info.get(
                    "camera_ocr_text"
                ),
                "camera_name_ocr_error": ocr_name_info.get(
                    "camera_ocr_error"
                ),
                "name_required_for_match": name_required,
                "codigo_representante_legal": ocr_name_info[
                    "codigo_representante_legal"
                ],
                "documents_found": len(documents),
                "pdfs_read": pdfs_read,
                "pages_with_signatures": pages_with_signatures,
                "signatures_compared": signatures_compared,
                "early_stop": False,
                "best_compared_candidate": summarize_attempt(best_attempt),
                "pdf_extraction": pdf_extraction_debug,
                "errors": errors,
                "debug_compare": best_attempt.get("debug_compare"),
            },
        }

        audit_payload = build_audit_payload(
            codigo_cliente=codigo_cliente,
            documents=documents,
            ocr_name_info=ocr_name_info,
            status="auto_match",
            match_automatico=True,
            best_attempt=best_attempt,
            errors=errors,
            audit_documents=audit_documents,
            condicion_entrega_id=condicion_entrega_id,
            fianza=fianza,
            camera_signature_image=camera_signature_crop,
        )

        return attach_audit(
            result,
            audit_payload,
            background_tasks=background_tasks,
        )

    result = {
        "ok": True,
        "match": False,
        "codigo_cliente": str(codigo_cliente),
        "condicion_entrega_id": condicion_entrega_id,
        "fianza": fianza,
        "message": "No se encontro coincidencia",
        "best_attempt": best_attempt,
        "images": {
            "camera_signature_base64": build_watermarked_signature_base64(
                camera_signature_crop
            ),
            "compared_document_signatures": compared_document_signatures,
        },
        "debug": {
            "camera_signatures_detected": len(camera_detections),
            "camera_detection": camera_detection_debug,
            "client_name": client_name,
            "client_names": client_names,
            "client_name_references": ocr_name_info.get(
                "name_references"
            ),
            "client_name_source": ocr_name_info["source"],
            "camera_name_ocr_text": ocr_name_info.get(
                "camera_ocr_text"
            ),
            "camera_name_ocr_error": ocr_name_info.get(
                "camera_ocr_error"
            ),
            "name_required_for_match": name_required,
            "codigo_representante_legal": ocr_name_info[
                "codigo_representante_legal"
            ],
            "documents_found": len(documents),
            "pdfs_read": pdfs_read,
            "pages_with_signatures": pages_with_signatures,
            "signatures_compared": signatures_compared,
            "early_stop": False,
            "best_compared_candidate": summarize_attempt(best_attempt),
            "pdf_extraction": pdf_extraction_debug,
            "errors": errors,
        },
    }

    audit_payload = build_audit_payload(
        codigo_cliente=codigo_cliente,
        documents=documents,
        ocr_name_info=ocr_name_info,
        status="no_match",
        match_automatico=False,
        best_attempt=best_attempt,
        errors=errors,
        audit_documents=audit_documents,
        condicion_entrega_id=condicion_entrega_id,
        fianza=fianza,
        camera_signature_image=camera_signature_crop,
    )

    return attach_audit(
        result,
        audit_payload,
        background_tasks=background_tasks,
    )
