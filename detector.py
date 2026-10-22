"""Local inference for the pinned SteganographIA image classifier."""

from io import BytesIO
import logging
import os
from threading import RLock

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "delpot/steganograph-ia-detector"
MODEL_REVISION = "b395557b96de82e5aaf97206054af416d657655a"
MODEL_NAME = f"{MODEL_ID}@{MODEL_REVISION[:8]}"
SCORE_MEANING = (
    "softmax-normalized model score; not calibrated as a probability of image origin"
)
LIMITATIONS = [
    "The model card reports testing on a balanced Defactify dataset; those results do not guarantee performance on other image sources.",
    "The model card reports that about 14% of AI-generated images in its test set were classified as real.",
    "The model card does not guarantee generalization to generators outside its training set, including Flux and newer generators.",
    "Images are resized to 224 × 224 pixels, which may remove high-resolution image artifacts.",
]

_model_lock = RLock()
_processor = None
_model = None
_device = None
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
if not any(
    getattr(handler, "_detector_console_handler", False)
    for handler in _LOGGER.handlers
):
    _console_handler = logging.StreamHandler()
    _console_handler._detector_console_handler = True
    _console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    _LOGGER.addHandler(_console_handler)
_LOGGER.propagate = False


class DetectorNotConfiguredError(RuntimeError):
    """Raised when no usable detection model is configured."""


class DetectorUnavailableError(RuntimeError):
    """Raised when the configured model cannot be loaded or used."""


def _load_model():
    global _device, _model, _processor

    if _model is not None and _processor is not None:
        return _processor, _model, _device

    local_files_only = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        processor = AutoImageProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_fast=False,
            local_files_only=local_files_only,
        )
        model, loading_info = AutoModelForImageClassification.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_safetensors=True,
            local_files_only=local_files_only,
            output_loading_info=True,
        )
        load_issues = {
            key: loading_info.get(key, [])
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
            if loading_info.get(key)
        }
        if load_issues:
            raise RuntimeError(
                f"The checkpoint did not load cleanly: {load_issues}"
            )

        _class_labels(model)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.to(device)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            logging.getLogger(__name__).warning(
                "Insufficient CUDA memory for the model; falling back to CPU."
            )
            model.to("cpu")
            torch.cuda.empty_cache()
            device = torch.device("cpu")
        _LOGGER.info(
            "Loaded pretrained checkpoint %s@%s; missing/unexpected/mismatched "
            "checkpoint keys: 0; model=%s; parameters=%d; device=%s",
            MODEL_ID,
            MODEL_REVISION,
            type(model).__name__,
            sum(parameter.numel() for parameter in model.parameters()),
            device,
        )
        _LOGGER.info(
            "Model class mapping from config: id2label=%s label2id=%s",
            model.config.id2label,
            model.config.label2id,
        )
        _LOGGER.info(
            "Image processor from checkpoint: %s",
            processor.to_dict(),
        )
    except Exception as error:
        if local_files_only:
            message = (
                "The pinned image model is not available in the local cache. "
                "Connect to the internet once to download it, then retry offline."
            )
        else:
            message = (
                "The pinned image model could not be downloaded or loaded. "
                "Check internet access, cached model files, and available memory."
            )
        raise DetectorUnavailableError(
            message
        ) from error

    _processor = processor
    _model = model
    _device = device
    return _processor, _model, _device


def _class_labels(model) -> dict[int, str]:
    configured_labels = model.config.id2label
    labels = {}
    for index, label in configured_labels.items():
        try:
            label_index = int(index)
        except (TypeError, ValueError):
            continue
        if isinstance(label, str):
            labels[label_index] = label

    if set(labels.values()) != {"real", "ai_generated"}:
        raise DetectorUnavailableError(
            "The loaded classifier does not provide the expected real and AI-generated labels."
        )
    if set(labels) != set(range(model.config.num_labels)):
        raise DetectorUnavailableError(
            "The classifier label indices do not match its configured output size."
        )

    configured_reverse = model.config.label2id
    reverse_labels = {}
    for label, index in configured_reverse.items():
        try:
            label_index = int(index)
        except (TypeError, ValueError):
            continue
        if isinstance(label, str):
            reverse_labels[label] = label_index
    expected_reverse = {label: index for index, label in labels.items()}
    if reverse_labels != expected_reverse:
        raise DetectorUnavailableError(
            "The classifier's id2label and label2id mappings are inconsistent."
        )
    return labels


def _predict(image: Image.Image, processor, model, device) -> dict[str, object]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise DetectorUnavailableError(
                "The classifier did not return tensor logits."
            )
        if (
            logits.ndim != 2
            or logits.shape[0] != 1
            or logits.shape[1] != model.config.num_labels
        ):
            raise DetectorUnavailableError(
                "The classifier output shape does not match its configured labels."
            )
        if not torch.isfinite(logits).all().item():
            raise DetectorUnavailableError(
                "The classifier returned non-finite output values."
            )
        probabilities = torch.softmax(logits, dim=-1)[0]

    labels = _class_labels(model)
    predicted_index = int(logits.argmax(dim=-1).item())
    label = labels.get(predicted_index)
    if label is None:
        raise DetectorUnavailableError(
            "The classifier returned a label not present in its configuration."
        )

    raw_logits = {
        labels[index]: float(logits[0, index].item())
        for index in range(model.config.num_labels)
    }
    class_scores = {
        labels[index]: float(probabilities[index].item())
        for index in range(model.config.num_labels)
    }
    _LOGGER.info(
        "Inference scores: raw_logits_by_class=%s softmax_probabilities_by_class=%s "
        "predicted_index=%d predicted_label=%s",
        raw_logits,
        class_scores,
        predicted_index,
        label,
    )

    return {
        "prediction_label": label,
        "score": float(probabilities[predicted_index].item()),
        "score_meaning": SCORE_MEANING,
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "class_scores": class_scores,
        "label_mapping": {
            str(index): labels[index] for index in range(model.config.num_labels)
        },
        "limitations": LIMITATIONS,
    }


def detect_image(image_data: bytes) -> dict[str, object]:
    """Classify decoded image bytes using the locally cached pretrained model."""
    global _device

    try:
        with Image.open(BytesIO(image_data)) as source:
            image = source.convert("RGB")

        try:
            with _model_lock:
                processor, model, device = _load_model()
                try:
                    return _predict(image, processor, model, device)
                except torch.cuda.OutOfMemoryError as error:
                    if device.type != "cuda":
                        raise DetectorUnavailableError(
                            "Image analysis ran out of available memory."
                        ) from error
                    logging.getLogger(__name__).warning(
                        "GPU memory was insufficient during inference; falling back to CPU."
                    )
                    model.to("cpu")
                    _device = torch.device("cpu")
                    torch.cuda.empty_cache()
                    return _predict(image, processor, model, _device)
        finally:
            image.close()
    except DetectorNotConfiguredError:
        raise
    except DetectorUnavailableError:
        raise
    except Exception as error:
        raise DetectorUnavailableError(
            "The local image model could not analyze this image."
        ) from error
