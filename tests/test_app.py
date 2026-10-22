"""Focused tests for upload validation, API errors, and detector contracts.

Test-only doubles are used only to trigger failure paths or verify label-index
mapping. They are never rendered as detector predictions.
"""

import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import app as app_module
import detector


def make_image(image_format="PNG", size=(16, 12), exif=None):
    buffer = io.BytesIO()
    image = Image.new("RGB", size, (80, 110, 140))
    save_options = {"exif": exif} if exif else {}
    image.save(buffer, format=image_format, **save_options)
    image.close()
    return buffer.getvalue()


class FlaskApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def post_image(self, content, filename="upload.png"):
        return self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(content), filename)},
            content_type="multipart/form-data",
        )

    def test_health_and_home_page(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json(), {"status": "ok"})

        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        self.assertIn('action="/api/analyze"', html)
        self.assertIn('id="image"', html)
        self.assertIn('image/jpeg,image/png,image/webp', html)
        self.assertEqual(self.client.get("/static/js/script.js").status_code, 200)
        self.assertEqual(self.client.get("/static/css/style.css").status_code, 200)

    def test_valid_jpeg_is_decoded_normalized_and_sent_to_inference(self):
        exif = Image.Exif()
        exif[274] = 6
        jpeg = make_image("JPEG", (16, 12), exif)
        observed = {}

        def detector_unavailable(normalized_image):
            with Image.open(io.BytesIO(normalized_image)) as normalized:
                observed["format"] = normalized.format
                observed["mode"] = normalized.mode
                observed["size"] = normalized.size
            return app_module._json_error(
                "model_unavailable", "Model unavailable in this test.", 503
            )

        with patch.object(app_module, "_run_inference", side_effect=detector_unavailable):
            response = self.post_image(jpeg, filename="misleading.webp")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error_code"], "model_unavailable")
        self.assertEqual(
            observed, {"format": "PNG", "mode": "RGB", "size": (12, 16)}
        )

    def test_missing_upload_returns_structured_error(self):
        response = self.client.post(
            "/api/analyze", data={}, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error_code"], "invalid_image")

    def test_unsupported_gif_is_rejected_from_decoded_format(self):
        response = self.post_image(make_image("GIF"), filename="valid.png")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error_code"], "invalid_image")

    def test_corrupted_png_is_rejected(self):
        corrupted = bytearray(make_image("PNG"))
        corrupted[-8] ^= 0xFF
        response = self.post_image(bytes(corrupted), filename="image.png")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error_code"], "invalid_image")

    def test_upload_over_10_mib_is_rejected(self):
        response = self.post_image(
            b"x" * (app_module.MAX_UPLOAD_SIZE + 1), filename="large.png"
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error_code"], "image_too_large")

    def test_inference_failure_returns_generic_error_without_exception_details(self):
        with patch.object(
            app_module,
            "detect_image",
            side_effect=RuntimeError("private test diagnostic"),
        ):
            response = self.post_image(make_image())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.get_json(),
            {
                "error": "The detector could not analyze this image.",
                "error_code": "inference_failed",
            },
        )
        self.assertNotIn("private test diagnostic", response.get_data(as_text=True))

    def test_unavailable_model_returns_safe_503_error(self):
        with patch.object(
            app_module,
            "detect_image",
            side_effect=detector.DetectorUnavailableError("private model location"),
        ):
            response = self.post_image(make_image())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json(),
            {
                "error": "The local image model could not be loaded or used.",
                "error_code": "model_unavailable",
            },
        )
        self.assertNotIn("private model location", response.get_data(as_text=True))

    def test_real_inference_response_schema_when_model_is_available(self):
        response = self.post_image(make_image())
        body = response.get_json()

        if response.status_code == 503:
            self.assertIn(
                body["error_code"], {"model_unavailable", "inference_busy"}
            )
            self.skipTest(
                "The real model is unavailable or busy; no test prediction was substituted."
            )

        self.assertEqual(response.status_code, 200, body)
        self.assertEqual(
            set(body),
            {
                "prediction_label",
                "score",
                "score_meaning",
                "class_scores",
                "label_mapping",
                "model_name",
                "model_id",
                "model_revision",
                "image_dimensions",
                "limitations",
            },
        )
        self.assertIn(body["prediction_label"], {"real", "ai_generated"})
        self.assertIsInstance(body["score"], float)
        self.assertIn("not calibrated", body["score_meaning"])
        self.assertEqual(set(body["class_scores"]), {"real", "ai_generated"})
        _, model, _ = detector._load_model()
        self.assertEqual(
            body["label_mapping"],
            {
                str(index): label
                for index, label in model.config.id2label.items()
            },
        )
        self.assertAlmostEqual(sum(body["class_scores"].values()), 1.0, places=6)
        self.assertEqual(
            body["prediction_label"],
            max(body["class_scores"], key=body["class_scores"].get),
        )
        self.assertEqual(body["score"], body["class_scores"][body["prediction_label"]])
        self.assertEqual(body["model_id"], detector.MODEL_ID)
        self.assertEqual(body["model_revision"], detector.MODEL_REVISION)
        self.assertIn("model_id", body)
        self.assertIn("model_revision", body)
        self.assertEqual(body["image_dimensions"], {"width": 16, "height": 12})
        self.assertIsInstance(body["limitations"], list)


class DetectorContractTests(unittest.TestCase):
    def test_label_indices_are_mapped_using_model_configuration(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={"1": "ai_generated", "0": "real"},
                label2id={"ai_generated": 1, "real": 0},
                num_labels=2,
            )
        )
        self.assertEqual(
            detector._class_labels(model),
            {0: "real", 1: "ai_generated"},
        )

    def test_reversed_label_indices_are_not_assumed(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "ai_generated", 1: "real"},
                label2id={"ai_generated": 0, "real": 1},
                num_labels=2,
            )
        )
        self.assertEqual(
            detector._class_labels(model),
            {0: "ai_generated", 1: "real"},
        )

    def test_missing_cached_model_files_raise_explicit_unavailable_error(self):
        original_state = (detector._processor, detector._model, detector._device)
        detector._processor = None
        detector._model = None
        detector._device = None
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            with patch.object(
                detector.AutoImageProcessor,
                "from_pretrained",
                side_effect=OSError("test missing processor files"),
            ):
                with self.assertRaises(detector.DetectorUnavailableError) as raised:
                    detector._load_model()
            self.assertIn("not available in the local cache", str(raised.exception))
        finally:
            detector._processor, detector._model, detector._device = original_state
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline


if __name__ == "__main__":
    unittest.main()
