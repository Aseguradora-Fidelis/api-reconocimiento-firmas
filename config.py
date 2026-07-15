import os
from dotenv import load_dotenv

load_dotenv()


def env_bool(name, default=False):
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "s",
        "si",
    }

# =========================================================
# GENERAL
# =========================================================
API_VERSION = "4.0.0"
DEBUG_SAVE_FILES = True
DEBUG_DIR = "debug_verify"

# =========================================================
# YOLO
# =========================================================
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "/home/thecouk/yolov8sFirmas.engine",
)

YOLO_CONF_THRESHOLD = float(
    os.getenv("YOLO_CONF_THRESHOLD", "0.15")
)

YOLO_IOU_THRESHOLD = float(
    os.getenv("YOLO_IOU_THRESHOLD", "0.70")
)

YOLO_MAX_DET = int(
    os.getenv("YOLO_MAX_DET", "20")
)

YOLO_IMGSZ = int(
    os.getenv("YOLO_IMGSZ", "320")
)

SIGNATURE_DETECTION_DEBUG = env_bool(
    "SIGNATURE_DETECTION_DEBUG",
    True,
)

SIGNATURE_DETECTION_DEBUG_DIR = os.getenv(
    "SIGNATURE_DETECTION_DEBUG_DIR",
    os.path.join(DEBUG_DIR, "detections"),
)

SIGNATURE_DEDUP_IOU_THRESHOLD = float(
    os.getenv("SIGNATURE_DEDUP_IOU_THRESHOLD", "0.45")
)

SIGNATURE_DEDUP_CONTAINMENT_THRESHOLD = float(
    os.getenv("SIGNATURE_DEDUP_CONTAINMENT_THRESHOLD", "0.75")
)

SIGNATURE_BEST_CONFIDENCE_RATIO = float(
    os.getenv("SIGNATURE_BEST_CONFIDENCE_RATIO", "0.90")
)

SIGNATURE_MIN_KEEP_CONFIDENCE = float(
    os.getenv("SIGNATURE_MIN_KEEP_CONFIDENCE", "0.60")
)

SIGNATURE_MAX_VARIANT_PIXELS = int(
    os.getenv("SIGNATURE_MAX_VARIANT_PIXELS", "12000000")
)

SIGNATURE_TILE_ROWS = int(
    os.getenv("SIGNATURE_TILE_ROWS", "3")
)

SIGNATURE_TILE_COLS = int(
    os.getenv("SIGNATURE_TILE_COLS", "2")
)

SIGNATURE_TILE_OVERLAP = float(
    os.getenv("SIGNATURE_TILE_OVERLAP", "0.12")
)

SIGNATURE_TILE_MIN_DIM = int(
    os.getenv("SIGNATURE_TILE_MIN_DIM", "900")
)

# En una foto de camara, los tiles pueden detectar distintas partes de una
# misma firma. Estas opciones permiten unir esas partes y conservar margen
# alrededor de los trazos antes de enviarlos al comparador.
CAMERA_SIGNATURE_MERGE_OVERLAP = float(
    os.getenv("CAMERA_SIGNATURE_MERGE_OVERLAP", "0.15")
)

CAMERA_SIGNATURE_MERGE_GAP_RATIO = float(
    os.getenv("CAMERA_SIGNATURE_MERGE_GAP_RATIO", "0.15")
)

CAMERA_SIGNATURE_PADDING_X = float(
    os.getenv("CAMERA_SIGNATURE_PADDING_X", "0.08")
)

CAMERA_SIGNATURE_PADDING_Y = float(
    os.getenv("CAMERA_SIGNATURE_PADDING_Y", "0.12")
)

# =========================================================
# COMPARE
# =========================================================
MATCH_THRESHOLD = float(
    os.getenv("MATCH_THRESHOLD", "0.30")
)

ALIGNED_WIDTH = 300
ALIGNED_HEIGHT = 150

MIN_COMPONENT_AREA = 12

SEARCH_SCALES = [0.88, 0.94, 1.00, 1.06, 1.12]

SEARCH_SHIFT_X = range(-24, 25, 4)

SEARCH_SHIFT_Y = range(-18, 19, 3)

SEARCH_ROTATIONS = [
    -8,
    -6,
    -4,
    -2,
    0,
    2,
    4,
    6,
    8,
]

# =========================================================
# PDF
# =========================================================
PDF_DPI = int(
    os.getenv("PDF_DPI", "300")
)

PDF_MAX_PAGES_TO_SCAN = int(
    os.getenv("PDF_MAX_PAGES_TO_SCAN", "10")
)

PDF_MAX_SIGNATURES_TO_COMPARE = int(
    os.getenv("PDF_MAX_SIGNATURES_TO_COMPARE", "6")
)

PDF_MAX_SIGNATURES_PER_PAGE = int(
    os.getenv("PDF_MAX_SIGNATURES_PER_PAGE", "4")
)

PDF_NAME_MATCH_THRESHOLD = float(
    os.getenv("PDF_NAME_MATCH_THRESHOLD", "0.60")
)

PDF_OCR_LANG = os.getenv(
    "PDF_OCR_LANG",
    "eng",
)

# =========================================================
# WATERMARK
# =========================================================
WATERMARK_TEXT = "FIDELIS"

# =========================================================
# AWS
# =========================================================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")

AWS_SECRET_ACCESS_KEY = os.getenv(
    "AWS_SECRET_ACCESS_KEY"
)

AWS_DEFAULT_BUCKET = os.getenv(
    "AWS_DEFAULT_BUCKET"
)

AWS_DEFAULT_REGION = os.getenv(
    "AWS_DEFAULT_REGION",
    "us-east-1",
)

S3_PRESIGNED_URL_EXPIRATION = int(
    os.getenv("S3_PRESIGNED_URL_EXPIRATION", "60")
)

S3_SIGNATURE_AUDIT_PREFIX = os.getenv(
    "S3_SIGNATURE_AUDIT_PREFIX",
    "signature-verification",
)

# =========================================================
# ORACLE
# =========================================================
DB_USERNAME = os.getenv("DB_USERNAME")

DB_PASSWORD = os.getenv("DB_PASSWORD")

DB_HOST_PRO = os.getenv("DB_HOST_PRO")

DB_SID_PRO = os.getenv("DB_SID_PRO")

ORACLE_CLIENT_PATH = os.getenv(
    "ORACLE_CLIENT_PATH",
    "/opt/oracle/instantclient",
)

# =========================================================
# VALIDATION
# =========================================================
def validate_required_config():
    required = {
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_DEFAULT_BUCKET": AWS_DEFAULT_BUCKET,
        "DB_USERNAME": DB_USERNAME,
        "DB_PASSWORD": DB_PASSWORD,
        "DB_HOST_PRO": DB_HOST_PRO,
        "DB_SID_PRO": DB_SID_PRO,
    }

    missing = [
        key
        for key, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Faltan variables de entorno: {', '.join(missing)}"
        )
