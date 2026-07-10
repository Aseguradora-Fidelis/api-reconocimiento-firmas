import logging

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


logger = logging.getLogger(__name__)

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

        return JSONResponse(result)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Error interno en /verify-signature")
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
    try:
        fecha_inicio = normalize_date_param(fecha_inicio)
        fecha_fin = normalize_date_param(fecha_fin)

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


@app.get("/verification-stats/daily")
def verification_stats_daily_endpoint(
    fecha_inicio: str,
    fecha_fin: str,
):
    try:
        fecha_inicio = normalize_date_param(fecha_inicio)
        fecha_fin = normalize_date_param(fecha_fin)

        return get_verification_stats_daily(
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

        return get_verifications(
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

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno consultando verificaciones: {str(e)}",
        )


@app.get("/verification/{verification_id}")
def verification_snapshot_endpoint(
    verification_id: int,
):
    try:
        snapshot = get_verification_snapshot(
            verification_id=verification_id,
        )

        if not snapshot:
            raise HTTPException(
                status_code=404,
                detail="Verificacion no encontrada",
            )

        return snapshot

    except HTTPException:
        raise

    except Exception as e:
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
            raise HTTPException(
                status_code=404,
                detail="Verificacion no encontrada",
            )

        snapshot = get_verification_snapshot(
            verification_id=verification_id,
        )

        return {
            "ok": True,
            "validation": validation,
            "snapshot": snapshot,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except HTTPException:
        raise

    except Exception as e:
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
    try:
        cliente = get_client_info(
            codigo_cliente=codigo_cliente,
        )

        if not cliente:
            raise HTTPException(
                status_code=404,
                detail="Cliente no encontrado",
            )

        return {
            "ok": True,
            "cliente": cliente,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno consultando cliente: {str(e)}",
        )

@app.get("/condicion-entrega/{condicion_entrega_id}")
def condicion_entrega_info_endpoint(
    condicion_entrega_id: int,
):
    try:
        info = get_condicion_entrega_info(
            condicion_entrega_id=condicion_entrega_id,
        )

        if not info:
            raise HTTPException(
                status_code=404,
                detail="Condición de entrega no encontrada",
            )

        return {
            "ok": True,
            "data": info,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}",
        )

@app.get("/poliza/{fianza}")
def poliza_info_endpoint(
    fianza: int,
):
    try:
        info = get_client_by_fianza(
            fianza=fianza,
        )

        if not info:
            raise HTTPException(
                status_code=404,
                detail="Póliza no encontrada",
            )

        return {
            "ok": True,
            "data": info,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}",
        )
