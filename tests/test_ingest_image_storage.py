from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised when Pillow is missing.
    Image = None

try:
    from narva_queue.service.ingest import _encode_resized_jpeg_bytes, _encode_resized_jpeg_from_path
except ImportError:  # pragma: no cover - exercised when SQLAlchemy/Pillow is missing.
    _encode_resized_jpeg_bytes = None
    _encode_resized_jpeg_from_path = None


@unittest.skipIf(
    Image is None or _encode_resized_jpeg_bytes is None or _encode_resized_jpeg_from_path is None,
    "Required dependencies are not installed in this interpreter",
)
class IngestImageStorageTests(unittest.TestCase):
    def test_encode_resized_jpeg_from_path_downscales_large_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jpg"
            Image.new("RGB", (1920, 1080), color=(20, 20, 20)).save(input_path, format="JPEG")
            output_bytes = _encode_resized_jpeg_from_path(
                input_path,
                max_width=960,
                max_height=540,
                quality=70,
            )

        with Image.open(BytesIO(output_bytes)) as output_image:
            self.assertEqual(output_image.format, "JPEG")
            self.assertEqual(output_image.size, (960, 540))

    def test_encode_resized_jpeg_bytes_keeps_smaller_image_size(self) -> None:
        image_bytes = BytesIO()
        Image.new("RGB", (400, 300), color=(50, 60, 70)).save(image_bytes, format="PNG")

        output_bytes = _encode_resized_jpeg_bytes(
            image_bytes.getvalue(),
            max_width=960,
            max_height=540,
            quality=70,
        )

        with Image.open(BytesIO(output_bytes)) as output_image:
            self.assertEqual(output_image.format, "JPEG")
            self.assertEqual(output_image.size, (400, 300))


if __name__ == "__main__":
    unittest.main()
