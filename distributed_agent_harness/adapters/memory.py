"""InMemoryNamespace — in-process document store for development and testing."""
from __future__ import annotations

import hashlib
import mimetypes
from typing import Union

from ..namespace import DocInfo, NamespaceAdapter


_StoredValue = Union[bytes, str]


class InMemoryNamespace(NamespaceAdapter):
    """Simple in-memory namespace adapter backed by a single dict.

    Stores text content as ``str`` and binary content as ``bytes``. Reads
    against the wrong kind (``read_doc`` on a binary path, or
    ``read_binary`` on a text path) return ``None`` rather than raising,
    matching the contract on ``NamespaceAdapter``: callers that need to
    disambiguate use ``doc_info``.

    State is lost when the process exits. Suitable for:
    - Local development
    - Unit and integration tests
    - Single-agent demos

    Not suitable for multi-process or multi-machine deployments — use a
    network-backed adapter (SharePoint, Google Drive, S3) for those.
    """

    def __init__(self) -> None:
        self._docs: dict[str, _StoredValue] = {}

    # ----------------------------------------------------------------------- #
    # Text                                                                     #
    # ----------------------------------------------------------------------- #

    def read_doc(self, path: str) -> str | None:
        value = self._docs.get(path)
        if isinstance(value, str):
            return value
        return None

    def write_doc(self, path: str, content: str) -> None:
        self._docs[path] = content

    # ----------------------------------------------------------------------- #
    # Binary                                                                   #
    # ----------------------------------------------------------------------- #

    def read_binary(self, path: str) -> bytes | None:
        value = self._docs.get(path)
        if isinstance(value, bytes):
            return value
        return None

    def write_binary(self, path: str, content: bytes) -> None:
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError(
                f"write_binary expects bytes, got {type(content).__name__}"
            )
        self._docs[path] = bytes(content)

    # ----------------------------------------------------------------------- #
    # Metadata + enumeration                                                   #
    # ----------------------------------------------------------------------- #

    def doc_info(self, path: str) -> DocInfo | None:
        value = self._docs.get(path)
        if value is None:
            return None
        if isinstance(value, str):
            return DocInfo(
                path=path,
                kind="text",
                size=len(value.encode("utf-8")),
                mime=_guess_mime(path) or "text/plain",
                sha256=hashlib.sha256(value.encode("utf-8")).hexdigest(),
            )
        return DocInfo(
            path=path,
            kind="binary",
            size=len(value),
            mime=_guess_mime(path),
            sha256=hashlib.sha256(value).hexdigest(),
        )

    def list_docs(self, prefix: str = "") -> list[str]:
        return sorted(p for p in self._docs if p.startswith(prefix))

    def delete_doc(self, path: str) -> None:
        self._docs.pop(path, None)

    def __repr__(self) -> str:
        return f"InMemoryNamespace(docs={list(self._docs.keys())})"


def _guess_mime(path: str) -> str | None:
    """Best-effort MIME type from the path extension."""
    mime, _encoding = mimetypes.guess_type(path)
    return mime
