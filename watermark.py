import base64
import cv2
import numpy as np

from config import WATERMARK_TEXT


WATERMARK_WIDTH = 420
WATERMARK_HEIGHT = 260


def image_to_base64(img: np.ndarray) -> str:
    success, buffer = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 84],
    )

    if not success:
        raise ValueError("No se pudo codificar imagen")

    return base64.b64encode(buffer).decode("utf-8")


def signature_to_visual_stroke(img: np.ndarray) -> np.ndarray:
    from compare_preprocess import preprocess_signature

    pre = preprocess_signature(img)
    stroke = pre["crop"]

    visual = 255 - stroke

    return cv2.cvtColor(
        visual,
        cv2.COLOR_GRAY2BGR,
    )


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("Imagen invalida para watermark")

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img.copy()


def fit_to_canvas(
    img: np.ndarray,
    width: int = WATERMARK_WIDTH,
    height: int = WATERMARK_HEIGHT,
) -> np.ndarray:
    img = ensure_bgr(img)

    h, w = img.shape[:2]

    scale = min(
        width / max(w, 1),
        height / max(h, 1),
    )

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def normalize_visual_base(img: np.ndarray) -> np.ndarray:
    img = fit_to_canvas(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.normalize(
        gray,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    faded = 150 + (gray.astype(np.float32) / 255.0) * 105

    faded = np.clip(
        faded,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        faded,
        cv2.COLOR_GRAY2BGR,
    )


def rotate_layer(layer: np.ndarray, angle: float):
    h, w = layer.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        layer,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def make_full_fidelis_layer(
    shape,
    text: str,
    angle: float,
    font_scale: float,
    thickness: int,
    step_x: int,
    step_y: int,
):
    h, w = shape[:2]

    layer = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    for y in range(-h, h * 2, step_y):
        for x in range(-w, w * 2, step_x):
            cv2.putText(
                layer,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return rotate_layer(
        layer,
        angle,
    )


def apply_black_text_layer(
    base: np.ndarray,
    layer: np.ndarray,
    strength: float,
) -> np.ndarray:
    mask = cv2.cvtColor(
        layer,
        cv2.COLOR_BGR2GRAY,
    )

    mask = mask.astype(np.float32) / 255.0
    mask = mask[:, :, None]

    result = base.astype(np.float32)
    result = result * (1.0 - mask * strength)

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def apply_watermark(
    img: np.ndarray,
    text: str = WATERMARK_TEXT,
) -> np.ndarray:
    base = normalize_visual_base(img)

    h, w = base.shape[:2]

    result = base.copy()

    # =====================================================
    # CAPA PRINCIPAL
    # =====================================================
    large_layer = make_full_fidelis_layer(
        shape=result.shape,
        text=text,
        angle=-28,
        font_scale=0.74,
        thickness=2,
        step_x=95,
        step_y=52,
    )

    result = apply_black_text_layer(
        result,
        large_layer,
        strength=0.78,
    )

    # =====================================================
    # CAPA CRUZADA
    # =====================================================
    cross_layer = make_full_fidelis_layer(
        shape=result.shape,
        text=text,
        angle=24,
        font_scale=0.58,
        thickness=1,
        step_x=105,
        step_y=58,
    )

    result = apply_black_text_layer(
        result,
        cross_layer,
        strength=0.58,
    )

    # =====================================================
    # CAPA HORIZONTAL
    # =====================================================
    horizontal_layer = make_full_fidelis_layer(
        shape=result.shape,
        text=text,
        angle=0,
        font_scale=0.42,
        thickness=1,
        step_x=82,
        step_y=46,
    )

    result = apply_black_text_layer(
        result,
        horizontal_layer,
        strength=0.42,
    )

    # =====================================================
    # MICRO WATERMARK
    # =====================================================
    micro_layer = make_full_fidelis_layer(
        shape=result.shape,
        text=text,
        angle=-12,
        font_scale=0.26,
        thickness=1,
        step_x=58,
        step_y=34,
    )

    result = apply_black_text_layer(
        result,
        micro_layer,
        strength=0.22,
    )

    # =====================================================
    # TEXTO CENTRAL
    # =====================================================
    center_layer = np.zeros_like(result)

    size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        2,
    )

    x = int((w - size[0]) / 2)
    y = int((h + size[1]) / 2)

    cv2.putText(
        center_layer,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    center_layer = rotate_layer(
        center_layer,
        -18,
    )

    result = apply_black_text_layer(
        result,
        center_layer,
        strength=0.70,
    )

    # =====================================================
    # FOOTER
    # =====================================================
    footer_h = 28

    cv2.rectangle(
        result,
        (0, h - footer_h),
        (w, h),
        (238, 238, 238),
        -1,
    )

    cv2.putText(
        result,
        f"{text} - USO RESTRINGIDO",
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # =====================================================
    # BORDE
    # =====================================================
    cv2.rectangle(
        result,
        (0, 0),
        (w - 1, h - 1),
        (0, 0, 0),
        1,
    )

    return result
def build_watermarked_base64(img: np.ndarray) -> str:
    marked = apply_watermark(img)
    return image_to_base64(marked)


def build_watermarked_signature_base64(img: np.ndarray) -> str:
    visual_stroke = signature_to_visual_stroke(img)
    marked = apply_watermark(visual_stroke)
    return image_to_base64(marked)