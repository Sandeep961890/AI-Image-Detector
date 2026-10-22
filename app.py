from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from io import BytesIO
import math
from threading import BoundedSemaphore
import warnings

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from detector import (
    DetectorNotConfiguredError,
    DetectorUnavailableError,
    detect_image,
)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
MAX_REQUEST_SIZE = MAX_UPLOAD_SIZE + 64 * 1024
MAX_IMAGE_PIXELS = 25_000_000
INFERENCE_TIMEOUT_SECONDS = 180
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_SIZE

_inference_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="image-inference"
)
_inference_slot = BoundedSemaphore(1)


class InvalidUploadError(ValueError):
    """Raised when uploaded content is not an acceptable, decodable image."""


class ImageTooLargeError(ValueError):
    """Raised when an image exceeds the supported decoded pixel limit."""


def _json_error(code: str, message: str, status: int):
    return jsonify({"error_code": code, "error": message}), status


def _read_upload() -> bytes:
    upload = request.files.get("image")
    if upload is None:
        raise InvalidUploadError("Upload an image using the 'image' field.")

    image_data = upload.stream.read(MAX_UPLOAD_SIZE + 1)
    if not image_data:
        raise InvalidUploadError("The uploaded image is empty.")
    if len(image_data) > MAX_UPLOAD_SIZE:
        raise ImageTooLargeError("The maximum upload size is 10 MB.")

    return image_data


def _prepare_detector_image(image_data: bytes) -> tuple[bytes, int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_data)) as source:
                if source.format not in SUPPORTED_FORMATS:
                    raise InvalidUploadError(
                        "Only JPEG, PNG, and WebP images are supported."
                    )
                source.verify()

            with Image.open(BytesIO(image_data)) as source:
                if source.format not in SUPPORTED_FORMATS:
                    raise InvalidUploadError(
                        "Only JPEG, PNG, and WebP images are supported."
                    )
                source.load()
                with ImageOps.exif_transpose(source) as oriented:
                    rgb_image = oriented.convert("RGB")
                    try:
                        width, height = rgb_image.size
                        if width * height > MAX_IMAGE_PIXELS:
                            raise ImageTooLargeError(
                                "The decoded image exceeds the supported pixel limit."
                            )

                        normalized = BytesIO()
                        rgb_image.save(normalized, format="PNG")
                        return normalized.getvalue(), width, height
                    finally:
                        rgb_image.close()
    except ImageTooLargeError:
        raise
    except Image.DecompressionBombError as error:
        raise ImageTooLargeError(
            "The decoded image exceeds the supported pixel limit."
        ) from error
    except Image.DecompressionBombWarning as error:
        raise ImageTooLargeError(
            "The decoded image exceeds the supported pixel limit."
        ) from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        if isinstance(error, InvalidUploadError):
            raise
        raise InvalidUploadError(
            "The uploaded file is not a valid JPEG, PNG, or WebP image."
        ) from error


def _run_inference(image_data: bytes):
    if not _inference_slot.acquire(blocking=False):
        return _json_error(
            "inference_busy",
            "The detector is still processing another image. Try again shortly.",
            503,
        )

    try:
        future = _inference_executor.submit(detect_image, image_data)
    except RuntimeError:
        _inference_slot.release()
        app.logger.exception("Could not submit image inference")
        return _json_error(
            "inference_unavailable", "The detector is temporarily unavailable.", 503
        )

    future.add_done_callback(lambda _future: _inference_slot.release())

    try:
        result = future.result(timeout=INFERENCE_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        return _json_error(
            "inference_timeout",
            "Image analysis exceeded the allowed processing time.",
            504,
        )
    except DetectorNotConfiguredError:
        return _json_error(
            "model_not_configured",
            "No AI detection model is configured. No prediction was made.",
            503,
        )
    except DetectorUnavailableError:
        app.logger.exception("The configured image model is unavailable")
        return _json_error(
            "model_unavailable",
            "The local image model could not be loaded or used.",
            503,
        )
    except Exception:
        app.logger.exception("Image inference failed")
        return _json_error(
            "inference_failed", "The detector could not analyze this image.", 500
        )

    if not isinstance(result, dict):
        app.logger.error("Detector returned an unexpected result type")
        return _json_error(
            "invalid_detector_response",
            "The detector returned an invalid result.",
            500,
        )

    response = {}
    prediction = result.get("prediction")
    if isinstance(result.get("prediction_label"), str):
        response["prediction_label"] = result["prediction_label"]
    elif isinstance(prediction, str):
        response["prediction_label"] = prediction
    elif isinstance(prediction, dict) and isinstance(prediction.get("label"), str):
        response["prediction_label"] = prediction["label"]
    elif isinstance(result.get("label"), str):
        response["prediction_label"] = result["label"]

    if "prediction_label" not in response:
        app.logger.error("Detector response did not include a prediction label")
        return _json_error(
            "invalid_detector_response",
            "The detector returned an invalid result.",
            500,
        )

    score = result.get("score")
    score_meaning = result.get("score_meaning")
    if isinstance(prediction, dict):
        score = prediction.get("score", score)
        score_meaning = prediction.get("score_meaning", score_meaning)
    if (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
        and isinstance(score_meaning, str)
        and score_meaning
    ):
        response["score"] = score
        response["score_meaning"] = score_meaning

    class_scores = result.get("class_scores")
    if isinstance(class_scores, dict) and all(
        isinstance(label, str)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for label, value in class_scores.items()
    ):
        response["class_scores"] = class_scores

    label_mapping = result.get("label_mapping")
    if isinstance(label_mapping, dict) and all(
        isinstance(index, str) and index.isdigit() and isinstance(label, str)
        for index, label in label_mapping.items()
    ):
        response["label_mapping"] = label_mapping

    model_name = result.get("model_name", result.get("model"))
    if isinstance(model_name, str):
        response["model_name"] = model_name

    for field in ("model_id", "model_revision"):
        if isinstance(result.get(field), str):
            response[field] = result[field]

    limitations = result.get("limitations")
    if isinstance(limitations, str):
        response["limitations"] = limitations
    elif isinstance(limitations, list) and all(
        isinstance(limitation, str) for limitation in limitations
    ):
        response["limitations"] = limitations

    return response


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/analyze")
def analyze():
    try:
        image_data = _read_upload()
        normalized_image, width, height = _prepare_detector_image(image_data)
    except ImageTooLargeError as error:
        return _json_error("image_too_large", str(error), 413)
    except InvalidUploadError as error:
        return _json_error("invalid_image", str(error), 400)

    result = _run_inference(normalized_image)
    if isinstance(result, tuple):
        return result

    result["image_dimensions"] = {"width": width, "height": height}
    return jsonify(result)


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_error):
    return _json_error(
        "request_too_large", "The maximum upload size is 10 MB.", 413
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1")
