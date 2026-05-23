"""Abstract base class for the Project Namespace storage adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod


class NamespaceAdapter(ABC):
    """
    Pluggable storage backend for the Project Namespace.

    All project state is stored as named documents organised under a project ID.
    Documents are human-readable text (JSON, Markdown, YAML) so that humans
    working in SharePoint, Google Drive, or a file browser can open and read them.

    Interface contract
    ------------------
    read_doc(path)         Return the document at path, or None if absent.
    write_doc(path, text)  Write (or overwrite) the document at path.
    list_docs(prefix)      Return all document paths, optionally filtered.

    Planned implementations
    -----------------------
    - InMemoryNamespace      built-in, for development and testing
    - SharePointNamespace    planned
    - GoogleDriveNamespace   planned
    - S3Namespace            planned
    """

    @abstractmethod
    def read_doc(self, path: str) -> str | None:
        """Return the document at *path*, or ``None`` if it does not exist."""

    @abstractmethod
    def write_doc(self, path: str, content: str) -> None:
        """Write *content* to *path*, creating the document if it does not exist."""

    @abstractmethod
    def list_docs(self, prefix: str = "") -> list[str]:
        """Return all document paths, optionally filtered to those starting with *prefix*."""

    def delete_doc(self, path: str) -> None:
        """Delete the document at *path*. Optional — not all backends require it."""
        raise NotImplementedError(f"{type(self).__name__} does not support delete_doc")
