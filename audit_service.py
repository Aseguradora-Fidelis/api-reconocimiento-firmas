import hashlib
import json
import logging
import uuid

import cv2
import oracledb

from config import (
    AWS_DEFAULT_BUCKET,
    S3_SIGNATURE_AUDIT_PREFIX,
)
from s3_oracle_service import (
    get_connection,
    s3_client,
)


logger = logging.getLogger(__name__)


BASE64_KEYS = {
    "camera_image_base64",
    "camera_signature_base64",
    "matched_document_signature_base64",
    "document_signature_base64",
}


def yes_no(value):
    return "S" if value else "N"


def safe_number(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truncate(value, max_length):
    if value is None:
        return None

    value = str(value)

    if len(value) <= max_length:
        return value

    return value[:max_length]


def sanitize_for_json(value):
    if isinstance(value, dict):
        sanitized = {}

        for key, item in value.items():
            if key in BASE64_KEYS:
                sanitized[key] = "[base64_omitted]"
                continue

            if key == "image":
                sanitized[key] = "[image_omitted]"
                continue

            if key in {"candidate_image", "context_image"}:
                sanitized[key] = "[image_omitted]"
                continue

            sanitized[key] = sanitize_for_json(item)

        return sanitized

    if isinstance(value, list):
        return [
            sanitize_for_json(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def json_dumps(value):
    if value is None:
        return None

    return json.dumps(
        sanitize_for_json(value),
        ensure_ascii=False,
        default=str,
    )


def insert_verification(cursor, payload, response):
    id_var = cursor.var(oracledb.NUMBER)

    cursor.execute(
        """
        INSERT INTO FIRMA_VERIFICACION (
            CODIGO_CLIENTE,
            CODIGO_REPRESENTANTE_LEGAL,
            NOMBRE_CLIENTE,
            NOMBRE_REPRESENTANTE_LEGAL,
            STATUS,
            MATCH_AUTOMATICO,
            BEST_SCORE,
            REQUEST_JSON,
            RESPONSE_JSON,
            ERROR_MESSAGE
        ) VALUES (
            :codigo_cliente,
            :codigo_representante_legal,
            :nombre_cliente,
            :nombre_representante_legal,
            :status,
            :match_automatico,
            :best_score,
            :request_json,
            :response_json,
            :error_message
        )
        RETURNING ID INTO :id
        """,
        {
            "codigo_cliente": payload.get("codigo_cliente"),
            "codigo_representante_legal": payload.get(
                "codigo_representante_legal"
            ),
            "nombre_cliente": truncate(
                payload.get("nombre_cliente"),
                300,
            ),
            "nombre_representante_legal": truncate(
                payload.get("nombre_representante_legal"),
                300,
            ),
            "status": payload.get("status"),
            "match_automatico": yes_no(payload.get("match_automatico")),
            "best_score": safe_number(payload.get("best_score")),
            "request_json": json_dumps(payload.get("request")),
            "response_json": json_dumps(response),
            "error_message": truncate(
                payload.get("error_message"),
                2000,
            ),
            "id": id_var,
        },
    )

    return int(id_var.getvalue()[0])


def insert_document(cursor, verification_id, document):
    debug = document.get("debug") or {}
    id_var = cursor.var(oracledb.NUMBER)

    cursor.execute(
        """
        INSERT INTO FIRMA_VERIFICACION_DOCUMENTO (
            VERIFICACION_ID,
            ARCHIVO,
            S3_KEY_ORIGINAL,
            TOTAL_PAGES,
            PAGES_SCANNED,
            SIGNATURES_DETECTED,
            SIGS_RETURNED_COMPARE,
            NAME_MATCHES,
            NAME_MISMATCHES,
            NAME_UNKNOWN,
            OCR_AVAILABLE,
            OCR_ATTEMPTS,
            OCR_ERRORS_JSON
        ) VALUES (
            :verification_id,
            :archivo,
            :s3_key_original,
            :total_pages,
            :pages_scanned,
            :signatures_detected,
            :sigs_returned_compare,
            :name_matches,
            :name_mismatches,
            :name_unknown,
            :ocr_available,
            :ocr_attempts,
            :ocr_errors_json
        )
        RETURNING ID INTO :id
        """,
        {
            "verification_id": verification_id,
            "archivo": truncate(document.get("archivo"), 500),
            "s3_key_original": truncate(document.get("s3_key"), 1000),
            "total_pages": debug.get("total_pages"),
            "pages_scanned": debug.get("pages_scanned"),
            "signatures_detected": debug.get("signatures_detected"),
            "sigs_returned_compare": debug.get(
                "signatures_returned_for_comparison"
            ),
            "name_matches": debug.get("name_matches"),
            "name_mismatches": debug.get("name_mismatches"),
            "name_unknown": debug.get("name_unknown"),
            "ocr_available": yes_no(debug.get("ocr_available")),
            "ocr_attempts": debug.get("ocr_attempts"),
            "ocr_errors_json": json_dumps(debug.get("ocr_errors")),
            "id": id_var,
        },
    )

    return int(id_var.getvalue()[0])


def insert_candidate(
    cursor,
    verification_id,
    document_id,
    candidate,
):
    name_match = candidate.get("name_match") or {}
    compare_result = candidate.get("compare_result") or {}
    id_var = cursor.var(oracledb.NUMBER)

    cursor.execute(
        """
        INSERT INTO FIRMA_VERIFICACION_CANDIDATO (
            VERIFICACION_ID,
            DOCUMENTO_ID,
            PAGE_NUMBER,
            SIGNATURE_INDEX,
            CANDIDATE_RANK,
            YOLO_CONFIDENCE,
            BBOX_JSON,
            CONTEXT_BBOX_JSON,
            OCR_TEXT,
            OCR_SOURCE,
            NAME_MATCH_STATUS,
            NAME_MATCH_SCORE,
            SCORE,
            DICE,
            IOU,
            THRESHOLD,
            MATCH_AUTOMATICO,
            COMPARED
        ) VALUES (
            :verification_id,
            :documento_id,
            :page_number,
            :signature_index,
            :candidate_rank,
            :yolo_confidence,
            :bbox_json,
            :context_bbox_json,
            :ocr_text,
            :ocr_source,
            :name_match_status,
            :name_match_score,
            :score,
            :dice,
            :iou,
            :threshold,
            :match_automatico,
            :compared
        )
        RETURNING ID INTO :id
        """,
        {
            "verification_id": verification_id,
            "documento_id": document_id,
            "page_number": candidate.get("page"),
            "signature_index": candidate.get("signature_index"),
            "candidate_rank": candidate.get("candidate_rank"),
            "yolo_confidence": safe_number(candidate.get("confidence")),
            "bbox_json": json_dumps(candidate.get("bbox")),
            "context_bbox_json": json_dumps(
                candidate.get("context_bbox")
            ),
            "ocr_text": candidate.get("name_text"),
            "ocr_source": candidate.get("name_text_source"),
            "name_match_status": name_match.get("status"),
            "name_match_score": safe_number(name_match.get("score")),
            "score": safe_number(compare_result.get("score")),
            "dice": safe_number(compare_result.get("dice")),
            "iou": safe_number(compare_result.get("iou")),
            "threshold": safe_number(compare_result.get("threshold")),
            "match_automatico": yes_no(compare_result.get("match")),
            "compared": yes_no(candidate.get("compared")),
            "id": id_var,
        },
    )

    return int(id_var.getvalue()[0])


def update_best_candidate(cursor, verification_id, best_candidate_id):
    if not best_candidate_id:
        return

    cursor.execute(
        """
        UPDATE FIRMA_VERIFICACION
           SET BEST_CANDIDATO_ID = :best_candidate_id,
               UPDATED_AT = SYSTIMESTAMP
         WHERE ID = :verification_id
        """,
        {
            "best_candidate_id": best_candidate_id,
            "verification_id": verification_id,
        },
    )


def encode_png(img):
    if img is None:
        return None

    ok, encoded = cv2.imencode(".png", img)

    if not ok:
        raise ValueError("No se pudo codificar imagen para auditoria")

    data = encoded.tobytes()
    height, width = img.shape[:2]

    return {
        "data": data,
        "content_type": "image/png",
        "width": int(width),
        "height": int(height),
        "file_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_image_s3_key(
    verification_id,
    tipo_imagen,
    candidate=None,
):
    prefix = S3_SIGNATURE_AUDIT_PREFIX.strip("/")
    image_uuid = uuid.uuid4().hex

    if candidate is None:
        return (
            f"{prefix}/{verification_id}/camera/"
            f"{image_uuid}_{tipo_imagen}.png"
        )

    candidate_id = candidate.get("_db_id")
    page = candidate.get("page")
    signature_index = candidate.get("signature_index")
    rank = candidate.get("candidate_rank")

    return (
        f"{prefix}/{verification_id}/candidates/"
        f"{image_uuid}_candidate_{candidate_id}_p{page}_f{signature_index}_"
        f"r{rank}_{tipo_imagen}.png"
    )


def upload_image_to_s3(s3_key, encoded):
    s3_client.put_object(
        Bucket=AWS_DEFAULT_BUCKET,
        Key=s3_key,
        Body=encoded["data"],
        ContentType=encoded["content_type"],
        Metadata={
            "sha256": encoded["sha256"],
        },
    )


def insert_image(
    cursor,
    verification_id,
    candidate_id,
    tipo_imagen,
    s3_key,
    encoded,
):
    id_var = cursor.var(oracledb.NUMBER)

    cursor.execute(
        """
        INSERT INTO FIRMA_VERIFICACION_IMAGEN (
            VERIFICACION_ID,
            CANDIDATO_ID,
            TIPO_IMAGEN,
            S3_KEY,
            CONTENT_TYPE,
            WIDTH,
            HEIGHT,
            FILE_SIZE,
            SHA256
        ) VALUES (
            :verification_id,
            :candidato_id,
            :tipo_imagen,
            :s3_key,
            :content_type,
            :width,
            :height,
            :file_size,
            :sha256
        )
        RETURNING ID INTO :id
        """,
        {
            "verification_id": verification_id,
            "candidato_id": candidate_id,
            "tipo_imagen": tipo_imagen,
            "s3_key": s3_key,
            "content_type": encoded["content_type"],
            "width": encoded["width"],
            "height": encoded["height"],
            "file_size": encoded["file_size"],
            "sha256": encoded["sha256"],
            "id": id_var,
        },
    )

    return int(id_var.getvalue()[0])


def persist_one_image(
    cursor,
    verification_id,
    candidate,
    tipo_imagen,
    img,
):
    encoded = encode_png(img)

    if encoded is None:
        return False

    s3_key = build_image_s3_key(
        verification_id,
        tipo_imagen,
        candidate=candidate,
    )

    upload_image_to_s3(
        s3_key,
        encoded,
    )

    insert_image(
        cursor=cursor,
        verification_id=verification_id,
        candidate_id=(
            candidate.get("_db_id")
            if candidate is not None
            else None
        ),
        tipo_imagen=tipo_imagen,
        s3_key=s3_key,
        encoded=encoded,
    )

    return True


def persist_audit_images(
    conn,
    cursor,
    verification_id,
    payload,
):
    image_count = 0

    try:
        if persist_one_image(
            cursor=cursor,
            verification_id=verification_id,
            candidate=None,
            tipo_imagen="camera_crop",
            img=payload.get("camera_signature_image"),
        ):
            image_count += 1

        for document in payload.get("documents") or []:
            for candidate in document.get("candidates") or []:
                if persist_one_image(
                    cursor=cursor,
                    verification_id=verification_id,
                    candidate=candidate,
                    tipo_imagen="candidate_crop",
                    img=candidate.get("candidate_image"),
                ):
                    image_count += 1

                if persist_one_image(
                    cursor=cursor,
                    verification_id=verification_id,
                    candidate=candidate,
                    tipo_imagen="context_crop",
                    img=candidate.get("context_image"),
                ):
                    image_count += 1

        conn.commit()

        return {
            "saved": True,
            "count": image_count,
        }

    except Exception as e:
        conn.rollback()
        logger.exception("Error guardando imagenes de auditoria")

        return {
            "saved": False,
            "count": image_count,
            "error": str(e),
        }


def persist_audit_images_task(verification_id, payload):
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        result = persist_audit_images(
            conn=conn,
            cursor=cursor,
            verification_id=verification_id,
            payload=payload,
        )

        if not result.get("saved"):
            logger.error(
                "No se guardaron imagenes de auditoria: %s",
                result,
            )

    except Exception:
        logger.exception(
            "Error inesperado en tarea background de imagenes"
        )

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()


def schedule_audit_images(
    background_tasks,
    verification_id,
    payload,
):
    if background_tasks is None:
        return False

    if not hasattr(background_tasks, "add_task"):
        return False

    background_tasks.add_task(
        persist_audit_images_task,
        verification_id,
        payload,
    )

    return True


def persist_verification_audit(
    payload,
    response,
    background_tasks=None,
):
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        verification_id = insert_verification(
            cursor,
            payload,
            response,
        )

        best_candidate_id = None

        for document in payload.get("documents") or []:
            document_id = insert_document(
                cursor,
                verification_id,
                document,
            )

            for candidate in document.get("candidates") or []:
                candidate_id = insert_candidate(
                    cursor,
                    verification_id,
                    document_id,
                    candidate,
                )

                if candidate.get("is_best"):
                    best_candidate_id = candidate_id

                candidate["_db_id"] = candidate_id

        update_best_candidate(
            cursor,
            verification_id,
            best_candidate_id,
        )

        conn.commit()

        if schedule_audit_images(
            background_tasks,
            verification_id,
            payload,
        ):
            return {
                "saved": True,
                "verification_id": verification_id,
                "images": {
                    "scheduled": True,
                    "saved": False,
                },
            }

        images_result = persist_audit_images(
            conn=conn,
            cursor=cursor,
            verification_id=verification_id,
            payload=payload,
        )

        return {
            "saved": True,
            "verification_id": verification_id,
            "images": images_result,
        }

    except Exception as e:
        if conn:
            conn.rollback()

        logger.exception("Error guardando auditoria de firmas")

        return {
            "saved": False,
            "error": str(e),
        }

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()
