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
