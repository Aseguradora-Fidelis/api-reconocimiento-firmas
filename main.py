import logging
from time import perf_counter

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Query,
    UploadFile,
    HTTPException,
)

from datetime import datetime

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    API_VERSION,
    validate_required_config,
)

from image_utils import (
    read_upload_file,
)

from verification_service import (
    verify_signature,
)

from verification_query_service import (
    deactivate_verification,
    get_verifications,
    get_verification_stats,
    get_verification_stats_daily,
    get_verification_snapshot,
    save_user_validation,
)

from s3_oracle_service import (
    get_client_info,
    get_condicion_entrega_info,
    get_client_by_fianza
)


# Uvicorn configura sus propios handlers y deja el logger raiz en WARNING.
# Usar este logger garantiza que los eventos INFO de negocio sean visibles
# junto al access log cuando la API se ejecuta con uvicorn.
logger = logging.getLogger("uvicorn.error")

# =========================================================
# VALIDATE CONFIG
# =========================================================
validate_required_config()

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="FIDELIS Signature API",
    version=API_VERSION,
)


class VerificationValidationRequest(BaseModel):
    candidate_id: int | None = None
    decision: str
    validated_by: str | None = None
    notes: str | None = None
    training_eligible: bool = False


def normalize_date_param(value: str):
    for date_format in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).strftime("%d/%m/%Y")
        except ValueError:
            pass

    raise ValueError("Las fechas deben tener formato DD/MM/YYYY o YYYY-MM-DD")


def parse_optional_int(value, field_name: str):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} debe ser numerico"
        ) from exc

# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporal para pruebas
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "FIDELIS Signature API",
        "version": API_VERSION,
    }

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "healthy",
    }

# =========================================================
# VERIFY
# =========================================================
@app.post("/verify-signature")
async def verify_signature_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    codigo_cliente: int = Form(...),
    condicion_entrega_id: str | None = Form(None),
    fianza: str | None = Form(None),
):
    started_at = perf_counter()
    logger.info(
        "verify-signature inicio codigo_cliente=%s "
        "condicion_entrega_id=%s fianza=%s content_type=%s",
        codigo_cliente,
        condicion_entrega_id,
        fianza,
        file.content_type,
    )

    try:
        camera_signature = read_upload_file(file)
        condicion_entrega_id = parse_optional_int(
            condicion_entrega_id,
            "condicion_entrega_id",
        )
        fianza = parse_optional_int(
            fianza,
            "fianza",
        )

        result = verify_signature(
            codigo_cliente=codigo_cliente,
            condicion_entrega_id=condicion_entrega_id,
            fianza=fianza,
            camera_signature=camera_signature,
            background_tasks=background_tasks,
        )

        debug = result.get("debug") or {}
        audit = debug.get("audit") or {}
        logger.info(
            "verify-signature fin codigo_cliente=%s ok=%s match=%s "
            "message=%r documents_found=%s pdfs_read=%s "
            "signatures_compared=%s errors=%s audit_saved=%s "
            "verification_id=%s duration_ms=%.1f",
            codigo_cliente,
            result.get("ok"),
            result.get("match"),
            result.get("message"),
            debug.get("documents_found"),
            debug.get("pdfs_read"),
            debug.get("signatures_compared"),
            len(debug.get("errors") or []),
            audit.get("saved"),
            audit.get("verification_id"),
            (perf_counter() - started_at) * 1000,
        )

        return JSONResponse(result)

    except ValueError as e:
        logger.warning(
            "verify-signature solicitud_invalida codigo_cliente=%s "
            "error=%s duration_ms=%.1f",
            codigo_cliente,
            e,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "verify-signature error_interno codigo_cliente=%s "
            "duration_ms=%.1f",
            codigo_cliente,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}",
        )

# =========================================================
# SAVED VERIFICATION
# =========================================================
@app.get("/verification-stats")
def verification_stats_endpoint(
    fecha_inicio: str,
    fecha_fin: str,
):
    started_at = perf_counter()
    logger.info(
        "verification-stats consulta fecha_inicio=%s fecha_fin=%s",
        fecha_inicio,
        fecha_fin,
    )

    try:
        fecha_inicio = normalize_date_param(fecha_inicio)
        fecha_fin = normalize_date_param(fecha_fin)

        result = get_verification_stats(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
        )
        logger.info(
            "verification-stats resultado fecha_inicio=%s fecha_fin=%s "
            "total_validaciones=%s clientes_distintos=%s errores=%s "
            "duration_ms=%.1f",
            fecha_inicio,
            fecha_fin,
            result.get("total_validaciones"),
            result.get("clientes_distintos"),
            result.get("errores"),
            (perf_counter() - started_at) * 1000,
        )
        return result

    except ValueError as e:
        logger.warning(
            "verification-stats parametros_invalidos error=%s "
            "duration_ms=%.1f",
            e,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        logger.exception(
            "verification-stats error fecha_inicio=%s fecha_fin=%s "
            "duration_ms=%.1f",
            fecha_inicio,
            fecha_fin,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno consultando estadisticas: {str(e)}",
        )


@app.get("/verification-stats/daily")
def verification_stats_daily_endpoint(
    fecha_inicio: str,
    fecha_fin: str,
):
    started_at = perf_counter()
    logger.info(
        "verification-stats-daily consulta fecha_inicio=%s fecha_fin=%s",
        fecha_inicio,
        fecha_fin,
    )

    try:
        fecha_inicio = normalize_date_param(fecha_inicio)
        fecha_fin = normalize_date_param(fecha_fin)

        result = get_verification_stats_daily(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
        )
        items = result.get("items") or []
        logger.info(
            "verification-stats-daily resultado fecha_inicio=%s "
            "fecha_fin=%s dias=%s total_validaciones=%s "
            "duration_ms=%.1f",
            fecha_inicio,
            fecha_fin,
            len(items),
            sum(item.get("total_validaciones") or 0 for item in items),
            (perf_counter() - started_at) * 1000,
        )
        return result

    except ValueError as e:
        logger.warning(
            "verification-stats-daily parametros_invalidos error=%s "
            "duration_ms=%.1f",
            e,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        logger.exception(
            "verification-stats-daily error fecha_inicio=%s fecha_fin=%s "
            "duration_ms=%.1f",
            fecha_inicio,
            fecha_fin,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno consultando estadisticas diarias: {str(e)}",
        )


@app.get("/reports/signature-validations/summary")
def signature_validations_summary_endpoint(
    start_date: str = Query(..., alias="startDate"),
    end_date: str = Query(..., alias="endDate"),
):
    try:
        fecha_inicio = normalize_date_param(start_date)
        fecha_fin = normalize_date_param(end_date)

        return get_verification_stats(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno consultando estadisticas: {str(e)}",
        )


@app.get("/verification")
def verifications_endpoint(
    fecha_inicio: str,
    fecha_fin: str,
    status: str | None = None,
    fecha: str | None = None,
    cliente: str | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1),
):
    started_at = perf_counter()
    logger.info(
        "verification listado consulta fecha_inicio=%s fecha_fin=%s "
        "page=%s page_size=%s",
        fecha_inicio,
        fecha_fin,
        page,
        page_size,
    )

    try:
        fecha_inicio = normalize_date_param(fecha_inicio)
        fecha_fin = normalize_date_param(fecha_fin)
        fecha = normalize_date_param(fecha) if fecha else None
        status = status.strip().lower() if status else None
        cliente = cliente.strip() if cliente else None

        if score_min is not None and score_max is not None:
            if score_min > score_max:
                raise ValueError(
                    "score_min no puede ser mayor que score_max"
                )

        result = get_verifications(
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            status=status,
            fecha=fecha,
            cliente=cliente,
            score_min=score_min,
            score_max=score_max,
            page=page,
            page_size=page_size,
        )
        logger.info(
            "verification listado total=%s items=%s page=%s page_size=%s "
            "status=%s fecha=%s cliente_filter=%s score_min=%s "
            "score_max=%s duration_ms=%.1f",
            result.get("total"),
            len(result.get("items") or []),
            page,
            page_size,
            status,
            fecha,
            bool(cliente),
            score_min,
            score_max,
            (perf_counter() - started_at) * 1000,
        )
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        logger.exception(
            "verification listado error duration_ms=%.1f",
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno consultando verificaciones: {str(e)}",
        )


@app.get("/verification/{verification_id}")
def verification_snapshot_endpoint(
    verification_id: int,
):
    started_at = perf_counter()
    logger.info(
        "verification detalle consulta verification_id=%s",
        verification_id,
    )

    try:
        snapshot = get_verification_snapshot(
            verification_id=verification_id,
        )

        if not snapshot:
            logger.warning(
                "verification detalle sin_resultado verification_id=%s "
                "duration_ms=%.1f",
                verification_id,
                (perf_counter() - started_at) * 1000,
            )
            raise HTTPException(
                status_code=404,
                detail="Verificacion no encontrada",
            )

        verification = snapshot.get("verification") or {}
        logger.info(
            "verification detalle encontrado verification_id=%s status=%s "
            "documents=%s images_ready=%s duration_ms=%.1f",
            verification_id,
            verification.get("status"),
            len(snapshot.get("documents") or []),
            (snapshot.get("images_status") or {}).get("ready"),
            (perf_counter() - started_at) * 1000,
        )
        return snapshot

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "verification detalle error verification_id=%s duration_ms=%.1f",
            verification_id,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno consultando verificacion: {str(e)}",
        )


@app.delete("/verification/{verification_id}")
def verification_delete_endpoint(
    verification_id: int,
):
    try:
        deleted = deactivate_verification(
            verification_id=verification_id,
        )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Verificacion no encontrada",
            )

        return {
            "ok": True,
            "verification_id": verification_id,
            "estado": "I",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno eliminando verificacion: {str(e)}",
        )


@app.post("/verification/{verification_id}/validate")
def verification_validate_endpoint(
    verification_id: int,
    payload: VerificationValidationRequest,
):
    started_at = perf_counter()
    logger.info(
        "verification validacion inicio verification_id=%s "
        "candidate_id=%s decision=%s training_eligible=%s",
        verification_id,
        payload.candidate_id,
        payload.decision,
        payload.training_eligible,
    )

    try:
        validation = save_user_validation(
            verification_id=verification_id,
            candidate_id=payload.candidate_id,
            decision=payload.decision,
            validated_by=payload.validated_by,
            notes=payload.notes,
            training_eligible=payload.training_eligible,
        )

        if not validation:
            logger.warning(
                "verification validacion sin_resultado verification_id=%s "
                "duration_ms=%.1f",
                verification_id,
                (perf_counter() - started_at) * 1000,
            )
            raise HTTPException(
                status_code=404,
                detail="Verificacion no encontrada",
            )

        snapshot = get_verification_snapshot(
            verification_id=verification_id,
        )

        logger.info(
            "verification validacion guardada verification_id=%s "
            "candidate_id=%s snapshot_found=%s duration_ms=%.1f",
            verification_id,
            payload.candidate_id,
            snapshot is not None,
            (perf_counter() - started_at) * 1000,
        )

        return {
            "ok": True,
            "validation": validation,
            "snapshot": snapshot,
        }

    except ValueError as e:
        logger.warning(
            "verification validacion invalida verification_id=%s "
            "error=%s duration_ms=%.1f",
            verification_id,
            e,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "verification validacion error verification_id=%s "
            "duration_ms=%.1f",
            verification_id,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno validando verificacion: {str(e)}",
        )

# =========================================================
# CLIENT INFO
# =========================================================
@app.get("/client/{codigo_cliente}")
def client_info_endpoint(
    codigo_cliente: int,
):
    started_at = perf_counter()
    logger.info("cliente consulta codigo_cliente=%s", codigo_cliente)

    try:
        cliente = get_client_info(
            codigo_cliente=codigo_cliente,
        )

        if not cliente:
            logger.warning(
                "cliente sin_resultado codigo_cliente=%s duration_ms=%.1f",
                codigo_cliente,
                (perf_counter() - started_at) * 1000,
            )
            raise HTTPException(
                status_code=404,
                detail="Cliente no encontrado",
            )

        logger.info(
            "cliente encontrado codigo_cliente=%s duration_ms=%.1f",
            codigo_cliente,
            (perf_counter() - started_at) * 1000,
        )
        return {
            "ok": True,
            "cliente": cliente,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "cliente error codigo_cliente=%s duration_ms=%.1f",
            codigo_cliente,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno consultando cliente: {str(e)}",
        )

@app.get("/condicion-entrega/{condicion_entrega_id}")
def condicion_entrega_info_endpoint(
    condicion_entrega_id: int,
):
    started_at = perf_counter()
    logger.info(
        "condicion-entrega consulta condicion_entrega_id=%s",
        condicion_entrega_id,
    )

    try:
        info = get_condicion_entrega_info(
            condicion_entrega_id=condicion_entrega_id,
        )

        if not info:
            logger.warning(
                "condicion-entrega sin_resultado condicion_entrega_id=%s "
                "duration_ms=%.1f",
                condicion_entrega_id,
                (perf_counter() - started_at) * 1000,
            )
            raise HTTPException(
                status_code=404,
                detail="Condición de entrega no encontrada",
            )

        logger.info(
            "condicion-entrega encontrada condicion_entrega_id=%s "
            "fianza=%s codigo_cliente=%s duration_ms=%.1f",
            condicion_entrega_id,
            info.get("fianza"),
            info.get("codigo_cliente"),
            (perf_counter() - started_at) * 1000,
        )
        return {
            "ok": True,
            "data": info,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "condicion-entrega error condicion_entrega_id=%s "
            "duration_ms=%.1f",
            condicion_entrega_id,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}",
        )

@app.get("/poliza/{fianza}")
def poliza_info_endpoint(
    fianza: int,
):
    started_at = perf_counter()
    logger.info("poliza consulta fianza=%s", fianza)

    try:
        info = get_client_by_fianza(
            fianza=fianza,
        )

        if not info:
            logger.warning(
                "poliza sin_resultado fianza=%s duration_ms=%.1f",
                fianza,
                (perf_counter() - started_at) * 1000,
            )
            raise HTTPException(
                status_code=404,
                detail="Póliza no encontrada",
            )

        logger.info(
            "poliza encontrada fianza=%s codigo_cliente=%s "
            "duration_ms=%.1f",
            fianza,
            info.get("codigo_cliente"),
            (perf_counter() - started_at) * 1000,
        )
        return {
            "ok": True,
            "data": info,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            "poliza error fianza=%s duration_ms=%.1f",
            fianza,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}",
        )
