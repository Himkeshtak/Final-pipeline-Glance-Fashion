"""Safe image discovery, raw-byte deduplication, decoding, and global resizing."""

from __future__ import annotations

import hashlib
import math
import tempfile
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

from fashion_indexer.config import DEFAULT_IMAGE_EXTENSIONS
from fashion_indexer.errors import DuplicateImageError, ImageDirectoryError, InvalidImageError

_HASH_CHUNK_SIZE = 1024 * 1024
_SPOOL_MEMORY_LIMIT = 16 * 1024 * 1024


@dataclass(slots=True)
class PreparedImage:
    """A decoded RGB image paired with its raw source identity."""

    path: Path
    file_hash: str
    image: Image.Image

    def close(self) -> None:
        self.image.close()


class ImagePreprocessor:
    """Discover and decode images while rejecting corrupt and duplicate inputs."""

    def __init__(
        self,
        images_dir: Path,
        *,
        recursive: bool = True,
        max_image_pixels: int = 100_000_000,
        max_image_dimension: int = 2_048,
        extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.recursive = recursive
        self.max_image_pixels = max_image_pixels
        self.max_image_dimension = max_image_dimension
        self.extensions = frozenset(self._normalize_extension(item) for item in extensions)
        self._seen_hashes: set[str] = set()

    def discover(self) -> list[Path]:
        """Return supported image files in deterministic path order."""

        if not self.images_dir.exists():
            raise ImageDirectoryError(
                f"Image directory does not exist: {self.images_dir.resolve(strict=False)}"
            )
        if not self.images_dir.is_dir():
            raise ImageDirectoryError(f"Image path is not a directory: {self.images_dir}")

        iterator = self.images_dir.rglob("*") if self.recursive else self.images_dir.glob("*")
        try:
            paths = [
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in self.extensions
            ]
        except OSError as exc:
            raise ImageDirectoryError(f"Cannot enumerate {self.images_dir}: {exc}") from exc
        return sorted(paths, key=lambda path: path.as_posix().casefold())

    def prepare(self, path: Path) -> PreparedImage:
        """Hash, deduplicate, validate, orient, convert, and safely resize one image."""

        source_path = Path(path)
        try:
            with (
                source_path.open("rb") as source,
                tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY_LIMIT, mode="w+b") as captured,
            ):
                digest = _new_md5()
                for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
                    captured.write(chunk)
                file_hash = digest.hexdigest()
                if file_hash in self._seen_hashes:
                    raise DuplicateImageError(
                        f"Duplicate raw image bytes for {source_path} (MD5 {file_hash})."
                    )

                captured.seek(0)
                image = self._decode_rgb(captured, source_path)
        except (DuplicateImageError, InvalidImageError):
            raise
        except (OSError, PermissionError) as exc:
            raise InvalidImageError(f"Cannot read image bytes from {source_path}: {exc}") from exc

        self._seen_hashes.add(file_hash)
        return PreparedImage(path=source_path.resolve(), file_hash=file_hash, image=image)

    def release_hash(self, file_hash: str) -> None:
        """Allow a later duplicate to retry after a per-image inference failure."""

        self._seen_hashes.discard(file_hash)

    @staticmethod
    def compute_md5(path: Path) -> str:
        """Compute the required MD5 over raw file bytes in bounded-memory chunks."""

        try:
            digest = _new_md5()
            with Path(path).open("rb") as source:
                for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except (OSError, PermissionError) as exc:
            raise InvalidImageError(f"Cannot read image bytes from {path}: {exc}") from exc

    def _decode_rgb(self, source_bytes: BinaryIO, path: Path) -> Image.Image:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(source_bytes) as probe:
                    self._validate_dimensions(path, probe.width, probe.height)
                    probe.verify()

                source_bytes.seek(0)
                with Image.open(source_bytes) as source:
                    if getattr(source, "n_frames", 1) > 1:
                        source.seek(0)
                    oriented = ImageOps.exif_transpose(source)
                    try:
                        oriented.load()
                        rgb = self._convert_to_rgb(oriented)
                    finally:
                        if oriented is not source:
                            oriented.close()

            return self._resize_if_needed(rgb)
        except InvalidImageError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
        ) as exc:
            raise InvalidImageError(f"Invalid or corrupt image {path}: {exc}") from exc

    def _validate_dimensions(self, path: Path, width: int, height: int) -> None:
        if width < 1 or height < 1:
            raise InvalidImageError(f"Image has invalid dimensions {width}x{height}: {path}")
        if width * height > self.max_image_pixels:
            raise InvalidImageError(
                f"Image exceeds safety limit of {self.max_image_pixels:,} pixels "
                f"({width}x{height}): {path}"
            )

    @staticmethod
    def _convert_to_rgb(image: Image.Image) -> Image.Image:
        has_alpha = image.mode in {"LA", "PA", "RGBA"} or "transparency" in image.info
        if has_alpha:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            try:
                composited = Image.alpha_composite(background, rgba)
                try:
                    return composited.convert("RGB")
                finally:
                    composited.close()
            finally:
                background.close()
                rgba.close()
        return image.convert("RGB")

    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        longest_edge = max(image.size)
        if longest_edge <= self.max_image_dimension:
            return image

        scale = self.max_image_dimension / longest_edge
        resized_size = (
            max(1, math.floor(image.width * scale)),
            max(1, math.floor(image.height * scale)),
        )
        try:
            return image.resize(resized_size, Image.Resampling.LANCZOS)
        finally:
            image.close()

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        normalized = extension.strip().lower()
        return normalized if normalized.startswith(".") else f".{normalized}"


def _new_md5() -> Any:
    try:
        return hashlib.md5(usedforsecurity=False)
    except TypeError:  # Python builds without the usedforsecurity keyword.
        return hashlib.md5()
