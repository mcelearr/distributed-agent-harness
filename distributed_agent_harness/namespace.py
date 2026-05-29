"""Abstract base class for the Project Namespace storage adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


# --------------------------------------------------------------------------- #
# DocInfo                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class DocInfo:
    """Metadata for one document in a namespace.

    Returned by ``NamespaceAdapter.doc_info``. Used by ``list_dir`` to
    populate file sizes (cheap on any adapter that tracks size natively,
    avoids reading the full content) and by the ``read`` meta-tool to
    decide whether to route through ``read_doc`` (text) or
    ``read_binary``.
    """
    path: str
    kind: Literal["text", "binary"]
    size: int
    #: Best-effort MIME type. ``None`` for adapters that don't track it.
    mime: str | None = None
    #: Hex SHA-256 of the content. ``None`` when the adapter does not
    #: compute hashes (e.g. when reading would be expensive).
    sha256: str | None = None


# --------------------------------------------------------------------------- #
# NamespaceAdapter                                                             #
# --------------------------------------------------------------------------- #

class NamespaceAdapter(ABC):
    """Pluggable storage backend for the Project Namespace.

    Documents are stored at hierarchical, slash-separated paths. The
    namespace presents a virtual filesystem to the agent:

    - **Text documents** (Markdown, JSON, plain text) are written via
      ``write_doc`` and read via ``read_doc``. ``read_doc`` returns
      ``None`` when no text document exists at the path (either because
      it's absent or because the path stores binary content).
    - **Binary documents** (PDFs, images, archives, spreadsheets) are
      written via ``write_binary`` and read via ``read_binary``. Both
      methods are optional — adapters that don't implement them raise
      ``NotImplementedError``.
    - ``doc_info`` reports metadata for either kind: size, MIME guess,
      and an optional content hash.
    - ``list_docs`` enumerates paths regardless of kind.

    Interface contract
    ------------------
    read_doc(path)          Return text content, or None.
    write_doc(path, text)   Write or overwrite text content.
    read_binary(path)       Return bytes, or None. Optional.
    write_binary(path, b)   Write or overwrite binary content. Optional.
    doc_info(path)          Return DocInfo, or None when no doc exists.
    list_docs(prefix)       Enumerate paths with a given prefix.
    delete_doc(path)        Remove a doc (text or binary). Optional.

    Planned implementations
    -----------------------
    - InMemoryNamespace      shipped, for development and testing
    - SharePointNamespace    planned
    - GoogleDriveNamespace   planned
    - S3Namespace            planned
    """

    @abstractmethod
    def read_doc(self, path: str) -> str | None:
        """Return the text document at *path*, or ``None``.

        ``None`` is returned both when no document exists at *path* and
        when the document at *path* is binary. Callers that need to
        disambiguate should consult ``doc_info``.
        """

    @abstractmethod
    def write_doc(self, path: str, content: str) -> None:
        """Write or overwrite *path* with text *content*."""

    @abstractmethod
    def list_docs(self, prefix: str = "") -> list[str]:
        """Return all document paths, optionally filtered by *prefix*."""

    def read_binary(self, path: str) -> bytes | None:
        """Return the binary content at *path*, or ``None`` if absent.

        Optional — adapters that don't support binary content raise
        ``NotImplementedError`` here.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support binary documents"
        )

    def write_binary(self, path: str, content: bytes) -> None:
        """Write or overwrite *path* with binary *content*.

        Optional — adapters that don't support binary content raise
        ``NotImplementedError`` here.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support binary documents"
        )

    def doc_info(self, path: str) -> DocInfo | None:
        """Return metadata for the document at *path*, or ``None``.

        Default implementation derives a best-effort answer from
        ``read_doc`` / ``read_binary`` so adapters get sensible behaviour
        for free. Adapters that natively track size, mime type, or
        modification times should override this to avoid the read.
        """
        text = self.read_doc(path)
        if text is not None:
            return DocInfo(
                path=path,
                kind="text",
                size=len(text.encode("utf-8")),
                mime="text/plain",
            )
        try:
            binary = self.read_binary(path)
        except NotImplementedError:
            return None
        if binary is None:
            return None
        return DocInfo(
            path=path,
            kind="binary",
            size=len(binary),
            mime=None,
        )

    def delete_doc(self, path: str) -> None:
        """Delete the document at *path*. Optional — adapters may decline."""
        raise NotImplementedError(f"{type(self).__name__} does not support delete_doc")
