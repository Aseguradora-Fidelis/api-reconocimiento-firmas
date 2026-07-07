import logging
import json

import oracledb

from image_utils import read_image_from_bytes
from s3_oracle_service import (
    get_connection,
    get_pdf_view_url,
    get_s3_object_bytes,
)
from watermark import (
    build_watermarked_base64,
    build_watermarked_signature_base64,
)


logger = logging.getLogger(__name__)

SIGNATURE_IMAGE_TYPES = {
    "camera_crop",
    "candidate_crop",
}


def read_lob(value):
    if value is None:
        return None

    if hasattr(value, "read"):
        return value.read()

    return value


def parse_json(value):
    value = read_lob(value)

    if not value:
        return None

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(value)
    except Exception:
        return value


def yes_no_to_bool(value):
    if value is None:
        return None

    return str(value).upper() == "S"


def serialize_datetime(value):
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def fetch_one_dict(cursor, query, params):
    cursor.execute(
        query,
        params,
    )

    row = cursor.fetchone()

    if not row:
        return None

    columns = [
        item[0].lower()
        for item in cursor.description
    ]

    return dict(zip(columns, row))


def fetch_all_dicts(cursor, query, params):
    cursor.execute(
        query,
        params,
    )

    columns = [
        item[0].lower()
        for item in cursor.description
    ]

    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]


def get_verification_stats(
    fecha_inicio: str,
    fecha_fin: str,
):
    default_stats = {
        "total_validaciones": 0,
        "clientes_distintos": 0,
        "auto_match": 0,
        "no_match": 0,
        "errores": 0,
        "pending_review": 0,
        "user_confirmed": 0,
        "user_rejected": 0,
        "only_accountant": 0,
    }

    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        row = fetch_one_dict(
            cursor,
            """
            SELECT
                COUNT(*) AS total_validaciones,
                COUNT(DISTINCT codigo_cliente) AS clientes_distintos,
                SUM(CASE WHEN status = 'auto_match' THEN 1 ELSE 0 END) AS auto_match,
                SUM(CASE WHEN status = 'no_match' THEN 1 ELSE 0 END) AS no_match,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errores,
                SUM(CASE WHEN status = 'pending_review' THEN 1 ELSE 0 END) AS pending_review,
                SUM(CASE WHEN status = 'user_confirmed' THEN 1 ELSE 0 END) AS user_confirmed,
                SUM(CASE WHEN status = 'user_rejected' THEN 1 ELSE 0 END) AS user_rejected,
                SUM(CASE WHEN status = 'only_accountant' THEN 1 ELSE 0 END) AS only_accountant
            FROM firma_verificacion
            WHERE TRUNC(created_at) BETWEEN
                  TO_DATE(:fecha_inicio, 'DD/MM/YYYY')
              AND TO_DATE(:fecha_fin, 'DD/MM/YYYY')
            """,
            {
                "fecha_inicio": fecha_inicio,
                "fecha_fin": fecha_fin,
            },
        )

        if not row:
            return default_stats

        return {
            key: int(row.get(key) or 0)
            for key in default_stats
        }

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()


def get_verification_stats_daily(
    fecha_inicio: str,
    fecha_fin: str,
):
    stats_keys = {
        "total_validaciones": 0,
        "clientes_distintos": 0,
        "auto_match": 0,
        "no_match": 0,
        "errores": 0,
        "pending_review": 0,
        "user_confirmed": 0,
        "user_rejected": 0,
        "only_accountant": 0,
    }

    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        rows = fetch_all_dicts(
            cursor,
            """
            SELECT
                TO_CHAR(TRUNC(created_at), 'YYYY-MM-DD') AS fecha,
                COUNT(*) AS total_validaciones,
                COUNT(DISTINCT codigo_cliente) AS clientes_distintos,
                SUM(CASE WHEN status = 'auto_match' THEN 1 ELSE 0 END) AS auto_match,
                SUM(CASE WHEN status = 'no_match' THEN 1 ELSE 0 END) AS no_match,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errores,
                SUM(CASE WHEN status = 'pending_review' THEN 1 ELSE 0 END) AS pending_review,
                SUM(CASE WHEN status = 'user_confirmed' THEN 1 ELSE 0 END) AS user_confirmed,
                SUM(CASE WHEN status = 'user_rejected' THEN 1 ELSE 0 END) AS user_rejected,
                SUM(CASE WHEN status = 'only_accountant' THEN 1 ELSE 0 END) AS only_accountant
            FROM firma_verificacion
            WHERE TRUNC(created_at) BETWEEN
                  TO_DATE(:fecha_inicio, 'DD/MM/YYYY')
              AND TO_DATE(:fecha_fin, 'DD/MM/YYYY')
            GROUP BY TRUNC(created_at)
            ORDER BY TRUNC(created_at)
            """,
            {
                "fecha_inicio": fecha_inicio,
                "fecha_fin": fecha_fin,
            },
        )

        return {
            "items": [
                {
                    "fecha": row["fecha"],
                    **{
                        key: int(row.get(key) or 0)
                        for key in stats_keys
                    },
                }
                for row in rows
            ]
        }

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()


def build_watermarked_s3_image_base64(
    s3_key,
    image_type,
):
    data = get_s3_object_bytes(s3_key)

    if not data:
        return None

    image = read_image_from_bytes(data)

    if image_type in SIGNATURE_IMAGE_TYPES:
        return build_watermarked_signature_base64(image)

    return build_watermarked_base64(image)


def build_image(row):
    s3_key = row["s3_key"]
    content_type = row["content_type"]
    image_type = row["tipo_imagen"]
    image_base64 = None

    try:
        image_base64 = build_watermarked_s3_image_base64(
            s3_key,
            image_type,
        )
    except Exception as exc:
        logger.exception(
            "Error preparando imagen de snapshot %s: %s",
            s3_key,
            exc,
        )

    return {
        "id": row["id"],
        "verification_id": row["verificacion_id"],
        "candidate_id": row["candidato_id"],
        "type": image_type,
        "image_base64": image_base64,
        "content_type": "image/jpeg",
        "watermarked": image_base64 is not None,
        "source_content_type": content_type,
        "source_width": row["width"],
        "source_height": row["height"],
        "source_file_size": row["file_size"],
        "source_sha256": row["sha256"],
        "created_at": serialize_datetime(row["created_at"]),
    }


def build_candidate(row):
    return {
        "id": row["id"],
        "document_id": row["documento_id"],
        "page": row["page_number"],
        "signature_index": row["signature_index"],
        "candidate_rank": row["candidate_rank"],
        "yolo_confidence": row["yolo_confidence"],
        "bbox": parse_json(row["bbox_json"]),
        "context_bbox": parse_json(row["context_bbox_json"]),
        "ocr_text": read_lob(row["ocr_text"]),
        "ocr_source": row["ocr_source"],
        "name_match": {
            "status": row["name_match_status"],
            "score": row["name_match_score"],
        },
        "comparison": {
            "compared": yes_no_to_bool(row["compared"]),
            "match": yes_no_to_bool(row["match_automatico"]),
            "score": row["score"],
            "dice": row["dice"],
            "iou": row["iou"],
            "threshold": row["threshold"],
        },
        "images": [],
        "created_at": serialize_datetime(row["created_at"]),
    }


def build_document(row):
    s3_key = row["s3_key_original"]

    return {
        "id": row["id"],
        "archivo": row["archivo"],
        "s3_key_original": s3_key,
        "file_view_url": get_pdf_view_url(s3_key),
        "stats": {
            "total_pages": row["total_pages"],
            "pages_scanned": row["pages_scanned"],
            "signatures_detected": row["signatures_detected"],
            "signatures_returned_for_comparison": row[
                "sigs_returned_compare"
            ],
            "name_matches": row["name_matches"],
            "name_mismatches": row["name_mismatches"],
            "name_unknown": row["name_unknown"],
            "ocr_available": yes_no_to_bool(row["ocr_available"]),
            "ocr_attempts": row["ocr_attempts"],
            "ocr_errors": parse_json(row["ocr_errors_json"]) or [],
        },
        "candidates": [],
        "created_at": serialize_datetime(row["created_at"]),
    }


def build_validation(row):
    if not row:
        return None

    return {
        "id": row["id"],
        "candidate_id": row["candidato_id"],
        "decision": row["decision"],
        "validated_by": row["validated_by"],
        "validated_at": serialize_datetime(row["validated_at"]),
        "notes": row["notes"],
        "training_eligible": yes_no_to_bool(row["training_eligible"]),
    }


def validate_decision(decision):
    allowed = {
        "confirmed",
        "rejected",
        "corrected",
        "only_accountant",
    }

    if decision not in allowed:
        raise ValueError(
            "Decision invalida. Use confirmed, rejected, corrected "
            "u only_accountant"
        )


def resolve_status_for_decision(decision):
    if decision == "confirmed":
        return "user_confirmed"

    if decision == "rejected":
        return "user_rejected"

    if decision == "only_accountant":
        return "only_accountant"

    return "corrected_by_user"


def get_candidate_for_validation(
    cursor,
    verification_id,
    candidate_id,
):
    return fetch_one_dict(
        cursor,
        """
        SELECT
            ID,
            VERIFICACION_ID
        FROM FIRMA_VERIFICACION_CANDIDATO
        WHERE ID = :candidate_id
          AND VERIFICACION_ID = :verification_id
        """,
        {
            "candidate_id": candidate_id,
            "verification_id": verification_id,
        },
    )


def is_required_candidate_error(exc):
    error = exc.args[0] if exc.args else exc
    code = getattr(error, "code", None)
    message = str(
        getattr(error, "message", None)
        or exc
    ).upper()

    return (
        code in {1400, 1407}
        and "CANDIDATO_ID" in message
    )


def save_user_validation(
    verification_id: int,
    candidate_id: int | None,
    decision: str,
    validated_by: str | None = None,
    notes: str | None = None,
    training_eligible: bool = False,
):
    decision = (decision or "").strip().lower()

    validate_decision(decision)

    if decision == "rejected":
        candidate_id = None

    if decision in {"confirmed", "corrected"} and not candidate_id:
        raise ValueError(
            "candidate_id es requerido para confirmed o corrected"
        )

    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        verification = fetch_one_dict(
            cursor,
            """
            SELECT ID
            FROM FIRMA_VERIFICACION
            WHERE ID = :verification_id
            """,
            {"verification_id": verification_id},
        )

        if not verification:
            return None

        if candidate_id:
            candidate = get_candidate_for_validation(
                cursor,
                verification_id,
                candidate_id,
            )

            if not candidate:
                raise ValueError(
                    "La candidata no pertenece a esta verificacion"
                )

        id_var = cursor.var(oracledb.NUMBER)

        cursor.execute(
            """
            INSERT INTO FIRMA_VALIDACION_USUARIO (
                VERIFICACION_ID,
                CANDIDATO_ID,
                DECISION,
                VALIDATED_BY,
                NOTES,
                TRAINING_ELIGIBLE
            ) VALUES (
                :verification_id,
                :candidate_id,
                :decision,
                :validated_by,
                :notes,
                :training_eligible
            )
            RETURNING ID INTO :id
            """,
            {
                "verification_id": verification_id,
                "candidate_id": candidate_id,
                "decision": decision,
                "validated_by": validated_by,
                "notes": notes,
                "training_eligible": "S" if training_eligible else "N",
                "id": id_var,
            },
        )

        status = resolve_status_for_decision(decision)

        if decision == "rejected":
            cursor.execute(
                """
                UPDATE FIRMA_VERIFICACION
                   SET STATUS = :status,
                       BEST_CANDIDATO_ID = NULL,
                       UPDATED_AT = SYSTIMESTAMP
                 WHERE ID = :verification_id
                """,
                {
                    "status": status,
                    "verification_id": verification_id,
                },
            )

        elif decision in {"confirmed", "corrected"}:
            cursor.execute(
                """
                UPDATE FIRMA_VERIFICACION
                   SET STATUS = :status,
                       BEST_CANDIDATO_ID = :candidate_id,
                       UPDATED_AT = SYSTIMESTAMP
                 WHERE ID = :verification_id
                """,
                {
                    "status": status,
                    "candidate_id": candidate_id,
                    "verification_id": verification_id,
                },
            )

        else:
            cursor.execute(
                """
                UPDATE FIRMA_VERIFICACION
                   SET STATUS = :status,
                       UPDATED_AT = SYSTIMESTAMP
                 WHERE ID = :verification_id
                """,
                {
                    "status": status,
                    "verification_id": verification_id,
                },
            )

        conn.commit()

        return {
            "validation_id": int(id_var.getvalue()[0]),
            "status": status,
        }

    except oracledb.DatabaseError as exc:
        if conn:
            conn.rollback()

        if (
            decision in {"rejected", "only_accountant"}
            and is_required_candidate_error(exc)
        ):
            raise ValueError(
                "La base de datos no permite registrar esta decision sin "
                "candidate_id. Permita NULL en "
                "FIRMA_VALIDACION_USUARIO.CANDIDATO_ID."
            ) from exc

        raise

    except Exception:
        if conn:
            conn.rollback()

        raise

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()


def get_verification_snapshot(verification_id: int):
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        verification = fetch_one_dict(
            cursor,
            """
            SELECT
                ID,
                MODELO_VERSION_ID,
                CODIGO_CLIENTE,
                CODIGO_REPRESENTANTE_LEGAL,
                NOMBRE_CLIENTE,
                NOMBRE_REPRESENTANTE_LEGAL,
                STATUS,
                MATCH_AUTOMATICO,
                BEST_SCORE,
                BEST_CANDIDATO_ID,
                REQUEST_JSON,
                RESPONSE_JSON,
                ERROR_MESSAGE,
                CREATED_AT,
                UPDATED_AT
            FROM FIRMA_VERIFICACION
            WHERE ID = :verification_id
            """,
            {"verification_id": verification_id},
        )

        if not verification:
            return None

        documents = fetch_all_dicts(
            cursor,
            """
            SELECT
                ID,
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
                OCR_ERRORS_JSON,
                CREATED_AT
            FROM FIRMA_VERIFICACION_DOCUMENTO
            WHERE VERIFICACION_ID = :verification_id
            ORDER BY ID
            """,
            {"verification_id": verification_id},
        )

        candidates = fetch_all_dicts(
            cursor,
            """
            SELECT
                ID,
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
                COMPARED,
                CREATED_AT
            FROM FIRMA_VERIFICACION_CANDIDATO
            WHERE VERIFICACION_ID = :verification_id
            ORDER BY DOCUMENTO_ID, CANDIDATE_RANK
            """,
            {"verification_id": verification_id},
        )

        images = fetch_all_dicts(
            cursor,
            """
            SELECT
                ID,
                VERIFICACION_ID,
                CANDIDATO_ID,
                TIPO_IMAGEN,
                S3_KEY,
                CONTENT_TYPE,
                WIDTH,
                HEIGHT,
                FILE_SIZE,
                SHA256,
                CREATED_AT
            FROM FIRMA_VERIFICACION_IMAGEN
            WHERE VERIFICACION_ID = :verification_id
            ORDER BY CANDIDATO_ID NULLS FIRST, TIPO_IMAGEN
            """,
            {"verification_id": verification_id},
        )

        validation = fetch_one_dict(
            cursor,
            """
            SELECT
                ID,
                VERIFICACION_ID,
                CANDIDATO_ID,
                DECISION,
                VALIDATED_BY,
                VALIDATED_AT,
                NOTES,
                TRAINING_ELIGIBLE
            FROM FIRMA_VALIDACION_USUARIO
            WHERE VERIFICACION_ID = :verification_id
            ORDER BY VALIDATED_AT DESC
            FETCH FIRST 1 ROW ONLY
            """,
            {"verification_id": verification_id},
        )

        document_map = {}
        candidate_map = {}

        built_documents = []
        for row in documents:
            document = build_document(row)
            document_map[document["id"]] = document
            built_documents.append(document)

        for row in candidates:
            candidate = build_candidate(row)
            candidate_map[candidate["id"]] = candidate

            document = document_map.get(candidate["document_id"])
            if document:
                document["candidates"].append(candidate)

        camera_images = []
        for row in images:
            image = build_image(row)
            candidate_id = image["candidate_id"]

            if candidate_id is None:
                camera_images.append(image)
                continue

            candidate = candidate_map.get(candidate_id)
            if candidate:
                candidate["images"].append(image)

        best_candidate_id = verification["best_candidato_id"]
        best_candidate = None

        if best_candidate_id:
            best_candidate = candidate_map.get(best_candidate_id)

        return {
            "ok": True,
            "verification": {
                "id": verification["id"],
                "modelo_version_id": verification["modelo_version_id"],
                "codigo_cliente": verification["codigo_cliente"],
                "codigo_representante_legal": verification[
                    "codigo_representante_legal"
                ],
                "nombre_cliente": verification["nombre_cliente"],
                "nombre_representante_legal": verification[
                    "nombre_representante_legal"
                ],
                "status": verification["status"],
                "match_automatico": yes_no_to_bool(
                    verification["match_automatico"]
                ),
                "best_score": verification["best_score"],
                "best_candidate_id": best_candidate_id,
                "error_message": verification["error_message"],
                "created_at": serialize_datetime(
                    verification["created_at"]
                ),
                "updated_at": serialize_datetime(
                    verification["updated_at"]
                ),
            },
            "request": parse_json(verification["request_json"]),
            "response": parse_json(verification["response_json"]),
            "camera_images": camera_images,
            "documents": built_documents,
            "best_candidate": best_candidate,
            "validation": build_validation(validation),
            "images_status": {
                "total": len(images),
                "ready": len(images) > 0,
            },
        }

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()
