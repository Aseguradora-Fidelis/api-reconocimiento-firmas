import os
from dotenv import load_dotenv

load_dotenv()

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
    os.getenv("YOLO_CONF_THRESHOLD", "0.10")
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
    os.getenv("PDF_DPI", "200")
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