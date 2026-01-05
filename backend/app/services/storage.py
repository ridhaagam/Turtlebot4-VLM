"""
Storage service for managing images and data persistence.
"""

import os
import base64
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from flask import current_app


class StorageService:
    """
    Service for managing file storage.
    """

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path or "/app/uploads")

    def save_image(
        self,
        image_data: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> str:
        """
        Save image data to disk.

        Args:
            image_data: Raw image bytes.
            filename: Name for the file.
            subfolder: Optional subfolder path.

        Returns:
            Full path to saved file.
        """
        # Create path with date-based organization
        if subfolder:
            folder = self.base_path / subfolder
        else:
            date_path = datetime.utcnow().strftime("%Y/%m/%d")
            folder = self.base_path / date_path

        folder.mkdir(parents=True, exist_ok=True)

        filepath = folder / filename

        with open(filepath, "wb") as f:
            f.write(image_data)

        return str(filepath)

    def save_base64_image(
        self,
        base64_data: str,
        filename: str,
        subfolder: Optional[str] = None
    ) -> str:
        """
        Save base64 encoded image to disk.

        Args:
            base64_data: Base64 encoded image string.
            filename: Name for the file.
            subfolder: Optional subfolder path.

        Returns:
            Full path to saved file.
        """
        image_data = base64.b64decode(base64_data)
        return self.save_image(image_data, filename, subfolder)

    def get_image(self, filepath: str) -> Optional[bytes]:
        """
        Read image data from disk.

        Args:
            filepath: Path to the image file.

        Returns:
            Image bytes or None if not found.
        """
        path = Path(filepath)
        if not path.exists():
            return None

        with open(path, "rb") as f:
            return f.read()

    def get_image_base64(self, filepath: str) -> Optional[str]:
        """
        Read image and return as base64 string.

        Args:
            filepath: Path to the image file.

        Returns:
            Base64 encoded string or None if not found.
        """
        image_data = self.get_image(filepath)
        if image_data:
            return base64.b64encode(image_data).decode("utf-8")
        return None

    def delete_image(self, filepath: str) -> bool:
        """
        Delete an image file.

        Args:
            filepath: Path to the image file.

        Returns:
            True if deleted successfully.
        """
        try:
            Path(filepath).unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def cleanup_old_images(self, days: int = 30) -> int:
        """
        Delete images older than specified days.

        Args:
            days: Delete images older than this many days.

        Returns:
            Number of files deleted.
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted = 0

        for filepath in self.base_path.rglob("*.jpg"):
            try:
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff:
                    filepath.unlink()
                    deleted += 1
            except Exception:
                continue

        # Clean up empty directories
        for dirpath in sorted(self.base_path.rglob("*"), reverse=True):
            if dirpath.is_dir() and not any(dirpath.iterdir()):
                try:
                    dirpath.rmdir()
                except Exception:
                    pass

        return deleted

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        total_files = 0
        total_size = 0

        for filepath in self.base_path.rglob("*.jpg"):
            total_files += 1
            total_size += filepath.stat().st_size

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_path": str(self.base_path),
        }
