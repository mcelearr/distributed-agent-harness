"""InMemoryNamespace — in-process document store for development and testing."""
from __future__ import annotations

from ..namespace import NamespaceAdapter


class InMemoryNamespace(NamespaceAdapter):
    """
    Simple in-memory namespace adapter backed by a plain Python dict.

    State is lost when the process exits. Suitable for:
    - Local development
    - Unit and integration tests
    - Single-agent demos

    Not suitable for multi-process or multi-machine deployments — use a
    network-backed adapter (SharePoint, Google Drive, S3) for those.
    """

    def __init__(self) -> None:
        self._docs: dict[str, str] = {}

    def read_doc(self, path: str) -> str | None:
        return self._docs.get(path)

    def write_doc(self, path: str, content: str) -> None:
        self._docs[path] = content

    def list_docs(self, prefix: str = "") -> list[str]:
        return sorted(p for p in self._docs if p.startswith(prefix))

    def delete_doc(self, path: str) -> None:
        self._docs.pop(path, None)

    def __repr__(self) -> str:
        return f"InMemoryNamespace(docs={list(self._docs.keys())})"
