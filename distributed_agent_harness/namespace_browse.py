"""
namespace_browse — read-only filesystem-style view over a ``NamespaceAdapter``.

The harness adopts a "filesystem-as-world" pattern (à la Claude Code / PI)
for the *read* side: the agent must be able to browse the project namespace
just like any other agent harness. The *write* side stays controlled —
mutations go through ``@actions`` only — so this module exposes nothing
that changes state.

Three operations:

- ``list_dir(adapter, path)`` — list entries under a namespace prefix.
  "Directories" are synthesised from common path prefixes in
  ``adapter.list_docs()`` results: if multiple docs share the prefix
  ``foo/bar/``, that's treated as a directory ``foo/bar/`` for browsing.
- ``read_doc(adapter, path, offset, limit)`` — read a single document with
  optional 1-indexed line offset/limit. Same shape as PI's ``read`` tool.
- ``grep_docs(adapter, pattern, path, glob, …)`` — substring/regex search
  across docs matching a path prefix and optional glob.

All three are pure functions over ``NamespaceAdapter``. They are reused by
the runtime (as built-in meta-tools the LLM sees), by future CLI
helpers, and by other agents inspecting the project over A2A — same
single-source-of-truth pattern as ``event_search``.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .namespace import NamespaceAdapter


# --------------------------------------------------------------------------- #
# Result shapes                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class DirEntry:
    """One row in ``list_dir`` output."""
    name: str
    kind: Literal["file", "directory"]
    size: int | None = None  # text length in chars; None for directories


@dataclass
class GrepMatch:
    """One match from ``grep_docs``."""
    path: str
    line_number: int   # 1-indexed
    line: str


# --------------------------------------------------------------------------- #
# list_dir                                                                     #
# --------------------------------------------------------------------------- #

def list_dir(adapter: "NamespaceAdapter", path: str = "") -> list[DirEntry]:
    """Return entries directly under ``path`` in the namespace.

    A directory is synthesised whenever multiple docs share a path prefix
    that contains more ``/`` separators. For example, with stored docs::

        demo/state.json
        demo/event_log.md
        demo/artefacts/legal_brief.pdf
        demo/artefacts/notes.txt

    ``list_dir(adapter, "demo/")`` returns::

        artefacts/   (directory)
        event_log.md (file)
        state.json   (file)

    Trailing ``/`` on ``path`` is optional and inserted if missing.
    """
    prefix = path
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    all_paths = adapter.list_docs(prefix=prefix)
    files: list[DirEntry] = []
    seen_dirs: set[str] = set()

    for full_path in all_paths:
        # Strip the prefix; what's left is the path relative to `path`.
        rel = full_path[len(prefix):] if prefix else full_path
        if not rel:
            # The doc lives exactly at the prefix path — uncommon but possible.
            continue
        if "/" in rel:
            # Nested. The first segment is a synthesised directory.
            dir_name = rel.split("/", 1)[0]
            seen_dirs.add(dir_name)
        else:
            content = adapter.read_doc(full_path)
            size = len(content) if content is not None else 0
            files.append(DirEntry(name=rel, kind="file", size=size))

    dirs = [DirEntry(name=d, kind="directory") for d in sorted(seen_dirs)]
    files.sort(key=lambda e: e.name)
    return dirs + files


def render_ls(entries: list[DirEntry]) -> str:
    """Human-readable rendering of ``list_dir`` output.

    Format matches Claude Code / PI conventions: directories first, then
    files alphabetically; directories carry a trailing ``/``.
    """
    if not entries:
        return "_(empty)_"
    lines: list[str] = []
    for e in entries:
        if e.kind == "directory":
            lines.append(f"{e.name}/")
        else:
            size_str = _format_size(e.size or 0)
            lines.append(f"{e.name}  ({size_str})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# read                                                                         #
# --------------------------------------------------------------------------- #

DEFAULT_READ_LIMIT = 2000  # lines, matches Claude Code / PI defaults


def read_doc(
    adapter: "NamespaceAdapter",
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> tuple[str | None, dict]:
    """Read a text doc with optional 1-indexed line offset/limit.

    Returns ``(content, meta)`` where ``meta`` includes ``total_lines`` and
    a ``truncated`` flag the runtime uses to decide whether to append a
    continuation hint.
    """
    raw = adapter.read_doc(path)
    if raw is None:
        return None, {}

    lines = raw.split("\n")
    total = len(lines)
    start = max(0, (offset - 1) if offset else 0)
    if start >= total:
        return "", {
            "total_lines": total,
            "truncated": False,
            "offset_out_of_range": True,
        }

    effective_limit = limit if limit is not None else DEFAULT_READ_LIMIT
    end = min(total, start + effective_limit)
    content = "\n".join(lines[start:end])
    return content, {
        "total_lines": total,
        "first_line": start + 1,
        "last_line": end,
        "truncated": end < total,
    }


def render_read(content: str | None, path: str, meta: dict) -> str:
    """Human-readable rendering for the LLM-facing TOOL message."""
    if content is None:
        return f"_(no such document: `{path}`)_"
    if meta.get("offset_out_of_range"):
        return (
            f"_(offset past end of file — `{path}` has "
            f"{meta.get('total_lines', 0)} lines)_"
        )
    if not meta.get("truncated"):
        return content
    return (
        f"{content}\n\n"
        f"[Showing lines {meta['first_line']}-{meta['last_line']} of "
        f"{meta['total_lines']}. Call `read` again with "
        f"`offset={meta['last_line'] + 1}` to continue.]"
    )


# --------------------------------------------------------------------------- #
# grep                                                                         #
# --------------------------------------------------------------------------- #

DEFAULT_GREP_LIMIT = 100


def grep_docs(
    adapter: "NamespaceAdapter",
    pattern: str,
    path: str = "",
    glob: str | None = None,
    ignore_case: bool = False,
    limit: int = DEFAULT_GREP_LIMIT,
) -> list[GrepMatch]:
    """Substring/regex search across docs under a prefix.

    The ``pattern`` is compiled as a regex (use ``re.escape`` upstream for
    literal-string searches). ``path`` filters the candidate set to docs
    whose full path starts with the given prefix; ``glob`` further filters
    by ``fnmatch``-style pattern over the full path.
    """
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error:
        # Fall back to literal substring search on bad regex.
        regex = re.compile(re.escape(pattern), flags)

    matches: list[GrepMatch] = []
    for full_path in adapter.list_docs(prefix=path):
        if glob is not None and not fnmatch.fnmatchcase(full_path, glob):
            continue
        content = adapter.read_doc(full_path)
        if not content:
            continue
        for lineno, line in enumerate(content.split("\n"), start=1):
            if regex.search(line):
                matches.append(GrepMatch(
                    path=full_path,
                    line_number=lineno,
                    line=line,
                ))
                if len(matches) >= limit:
                    return matches
    return matches


def render_grep(matches: list[GrepMatch], limit: int = DEFAULT_GREP_LIMIT) -> str:
    """Human-readable rendering of grep results."""
    if not matches:
        return "_(no matches)_"
    lines = [f"{m.path}:{m.line_number}: {m.line}" for m in matches]
    if len(matches) >= limit:
        lines.append(f"\n[Hit limit={limit}; raise `limit` to see more matches.]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _format_size(n: int) -> str:
    """Compact size formatter (B / KB / MB)."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"
