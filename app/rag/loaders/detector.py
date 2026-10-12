import mimetypes
from pathlib import Path


class FileTypeDetector:
    """
    Detect supported document types using file extension and MIME type.
    """

    SUPPORTED_TYPES = {
        ".pdf": "pdf",
        ".txt": "text",
        ".text": "text",
        ".csv": "csv",
        ".md": "text",
        ".markdown": "text",
    }

    def detect(self, file_path: str | Path) -> str:
        path = Path(file_path)

        extension = path.suffix.lower()

        file_type = self.SUPPORTED_TYPES.get(extension)

        if file_type:
            return file_type

        mime_type, _ = mimetypes.guess_type(str(path))

        if mime_type == "application/pdf":
            return "pdf"

        if mime_type == "text/plain":
            return "text"

        if mime_type == "text/csv":
            return "csv"

        raise ValueError(
            f"Unsupported file type: {extension or mime_type}"
        )