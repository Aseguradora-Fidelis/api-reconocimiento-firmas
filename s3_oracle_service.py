import logging
from io import BytesIO

import boto3
import oracledb

from botocore.exceptions import ClientError

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_BUCKET,
    AWS_DEFAULT_REGION,
    S3_PRESIGNED_URL_EXPIRATION,
    DB_USERNAME,
    DB_PASSWORD,
    DB_HOST_PRO,
    DB_SID_PRO,
    ORACLE_CLIENT_PATH,
)

# =========================================================
# LOGGING
# =========================================================
logger = logging.getLogger(__name__)

# =========================================================
# ORACLE INIT
# =========================================================
oracledb.init_oracle_client(
    lib_dir=ORACLE_CLIENT_PATH,
)

# =========================================================
# S3
# =========================================================
s3_client = boto3.client(
    "s3",
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# =========================================================
# ORACLE
# =========================================================
def get_connection():
    dsn = oracledb.makedsn(
        host=DB_HOST_PRO,
        port=1521,
        service_name=DB_SID_PRO,
    )

    return oracledb.connect(
        user=DB_USERNAME,
        password=DB_PASSWORD,
        dsn=dsn,
    )


# =========================================================
# CLIENT DOCUMENTS
# =========================================================
def get_client_documents(
    codigo_cliente: int,
):
    conn = None
    cursor = None

    try:
        conn = get_connection()

        cursor = conn.cursor()

        query = """
            SELECT
                DOCS.DESCRIPCION,
                DOCS.S3_KEY,
                DOCS.CODIGO_CLIENTE,
                main_fusa.pkg_general.nombre_contacto(
                    DOCS.CODIGO_CLIENTE
                ) AS nombre_cliente,
                DOCS.CODIGO_REPRESENTANTE_LEGAL,
                main_fusa.pkg_general.nombre_contacto(
                    DOCS.CODIGO_REPRESENTANTE_LEGAL
                ) AS nombre_representante_legal
            FROM (
                SELECT
                    ADJ.DESCRIPCION,
                    BLB."KEY" AS S3_KEY,
                    CLI.COD_CONTACTO AS CODIGO_CLIENTE,
                    main_fusa.pkg_doc_cg.obtener_representante_legal(
                        NULL,
                        CLI.COD_CONTACTO
                    ) AS CODIGO_REPRESENTANTE_LEGAL
                FROM MAIN_FUSA.MG_CONTACTOS CLI

                JOIN CONTACTO_EXPEDIENTE EX
                    ON CLI.COD_CONTACTO = EX.COD_CONTACTO

                JOIN CONTACTO_EXPEDIENTE_ARCHIVO ARCH
                    ON EX.ID = ARCH.CONTACTO_EXPEDIENTE_ID

                JOIN CONTACTO_ARCHIVOS_ADJUNTOS ADJ
                    ON ARCH.ADJUNTO_ID = ADJ.ID

                JOIN TIPO_ARCHIVO_ADJUNTO TADJ
                    ON ARCH.TIPO_ARCHIVO_ADJUNTO_ID = TADJ.ID

                JOIN ACTIVE_STORAGE_ATTACHMENTS ATC
                    ON ARCH.ID = ATC.RECORD_ID

                JOIN ACTIVE_STORAGE_BLOBS BLB
                    ON ATC.BLOB_ID = BLB.ID

                WHERE CLI.COD_CONTACTO = :codigo_cliente
                  AND LOWER(TADJ.DESCRIPCION) LIKE  LOWER('%Estados%financieros%')
                  AND ATC.NAME = 'archivo_verificado'
            ) DOCS
        """

        cursor.execute(
            query,
            {"codigo_cliente": codigo_cliente},
        )

        rows = cursor.fetchall()

        documents = []

        for row in rows:
            documents.append({
                "archivo": row[0],
                "s3_key": row[1],
                "codigo_cliente": row[2],
                "nombre_cliente": row[3],
                "codigo_representante_legal": row[4],
                "nombre_representante_legal": row[5],
            })

        return documents

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()


# =========================================================
# S3 PDF MEMORY
# =========================================================
def get_s3_object_bytes(
    s3_key: str,
):
    try:
        response = s3_client.get_object(
            Bucket=AWS_DEFAULT_BUCKET,
            Key=s3_key,
        )

        return response["Body"].read()

    except ClientError as e:
        logger.error(
            f"S3 error leyendo {s3_key}: {e}"
        )

    except Exception as e:
        logger.error(
            f"Error leyendo archivo S3 {s3_key}: {e}"
        )

    return None


def get_pdf_from_s3(
    s3_key: str,
):
    data = get_s3_object_bytes(s3_key)

    if data is None:
        return None

    return BytesIO(data)


def get_pdf_view_url(
    s3_key: str,
):
    try:
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": AWS_DEFAULT_BUCKET,
                "Key": s3_key,
                "ResponseContentType": "application/pdf",
                "ResponseContentDisposition": "inline",
            },
            ExpiresIn=S3_PRESIGNED_URL_EXPIRATION,
        )

    except Exception as e:
        logger.error(
            f"Error generando URL de visualizacion para {s3_key}: {e}"
        )

    return None


def get_s3_view_url(
    s3_key: str,
    content_type: str | None = None,
):
    try:
        params = {
            "Bucket": AWS_DEFAULT_BUCKET,
            "Key": s3_key,
            "ResponseContentDisposition": "inline",
        }

        if content_type:
            params["ResponseContentType"] = content_type

        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params=params,
            ExpiresIn=S3_PRESIGNED_URL_EXPIRATION,
        )

    except Exception as e:
        logger.error(
            f"Error generando URL S3 para {s3_key}: {e}"
        )

    return None

# =========================================================
# CLIENT INFO
# =========================================================
def get_client_info(
    codigo_cliente: int,
):
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                CLI.COD_CONTACTO,
                main_fusa.pkg_general.nombre_contacto(CLI.COD_CONTACTO) AS nombre_cliente
            FROM MAIN_FUSA.MG_CONTACTOS CLI
            WHERE CLI.COD_CONTACTO = :codigo_cliente
        """

        cursor.execute(
            query,
            {"codigo_cliente": codigo_cliente},
        )

        row = cursor.fetchone()

        if not row:
            return None

        return {
            "codigo_cliente": row[0],
            "nombre": row[1]
        }

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()

def get_condicion_entrega_info(
    condicion_entrega_id: int,
):
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                ce.id,
                ce.fianza,
                fp.COD_CONTACTO_CLIENTE,
                main_fusa.pkg_general.nombre_contacto(fp.COD_CONTACTO_CLIENTE) AS nombre_cliente
            FROM
                fia_condicion_entrega ce,
                FIA_POLIZA fp
            WHERE
                ce.fianza = fp.fianza
                AND ce.id = :id
        """

        cursor.execute(
            query,
            {"id": condicion_entrega_id},
        )

        row = cursor.fetchone()

        if not row:
            return None

        return {
            "condicion_entrega_id": row[0],
            "fianza": row[1],
            "codigo_cliente": row[2],
            "nombre_cliente": row[3],
        }

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()

def get_client_by_fianza(
    fianza: int,
):
    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                P.FIANZA,
                P.COD_CONTACTO_CLIENTE,
                main_fusa.pkg_general.nombre_contacto(P.COD_CONTACTO_CLIENTE) AS nombre_cliente
            FROM MAIN_FUSA.FIA_POLIZA P
            WHERE P.FIANZA = :fianza
        """

        cursor.execute(
            query,
            {"fianza": fianza},
        )

        row = cursor.fetchone()

        if not row:
            return None

        return {
            "fianza": row[0],
            "codigo_cliente": row[1],
            "nombre": row[2],
        }

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()
