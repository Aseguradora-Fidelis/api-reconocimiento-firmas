from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
)

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
    get_verification_snapshot,
    save_user_validation,
)

from s3_oracle_service import (
    get_client_info,
    get_condicion_entrega_info,
    get_client_by_fianza
)

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
):
    try:
        camera_signature = read_upload_file(file)

        result = verify_signature(
            codigo_cliente=codigo_cliente,
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
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}",
        )

# =========================================================
# SAVED VERIFICATION
# =========================================================
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
