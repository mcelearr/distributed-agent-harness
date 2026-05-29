"""Tests for namespace_browse — the read-only filesystem-style view over
``NamespaceAdapter``. Covers list_dir / read_doc / grep_docs and the
markdown renderers.
"""
from __future__ import annotations

import pytest

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.namespace_browse import (
    DirEntry,
    grep_docs,
    list_dir,
    read_doc,
    render_grep,
    render_ls,
    render_read,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def ns() -> InMemoryNamespace:
    """A namespace with a realistic mix of docs and subdirectories."""
    n = InMemoryNamespace()
    n.write_doc("demo/state.json", '{"counter": 7}')
    n.write_doc("demo/summary.md", "# Demo\n\nProject summary.")
    n.write_doc(
        "demo/event_log.md",
        "# Event Log\n\n- offset 0: add_item\n- offset 1: consult_legal\n",
    )
    n.write_doc("demo/artefacts/legal_brief.md", "Brief about Acme Corp.\nLine 2.")
    n.write_doc("demo/artefacts/notes.txt", "Note about Acme.\nUnrelated line.")
    n.write_doc("other/state.json", '{"different": "project"}')
    return n


# --------------------------------------------------------------------------- #
# list_dir                                                                     #
# --------------------------------------------------------------------------- #

class TestListDir:
    def test_root_lists_top_level(self, ns: InMemoryNamespace) -> None:
        entries = list_dir(ns, "")
        names = [(e.name, e.kind) for e in entries]
        # Both top-level "projects" are seen as directories
        assert ("demo", "directory") in names
        assert ("other", "directory") in names

    def test_project_root_lists_files_and_subdirs(self, ns: InMemoryNamespace) -> None:
        entries = list_dir(ns, "demo/")
        kinds = {(e.name, e.kind) for e in entries}
        # The synthetic directory comes through
        assert ("artefacts", "directory") in kinds
        # Files at the project root come through
        assert ("state.json", "file") in kinds
        assert ("summary.md", "file") in kinds
        assert ("event_log.md", "file") in kinds

    def test_directories_appear_before_files(self, ns: InMemoryNamespace) -> None:
        entries = list_dir(ns, "demo/")
        # All dirs precede all files in the output
        last_dir_idx = max(
            (i for i, e in enumerate(entries) if e.kind == "directory"),
            default=-1,
        )
        first_file_idx = next(
            (i for i, e in enumerate(entries) if e.kind == "file"),
            len(entries),
        )
        assert last_dir_idx < first_file_idx

    def test_trailing_slash_is_optional(self, ns: InMemoryNamespace) -> None:
        with_slash = list_dir(ns, "demo/")
        without_slash = list_dir(ns, "demo")
        assert {(e.name, e.kind) for e in with_slash} == {
            (e.name, e.kind) for e in without_slash
        }

    def test_files_carry_size(self, ns: InMemoryNamespace) -> None:
        entries = list_dir(ns, "demo/")
        state = next(e for e in entries if e.name == "state.json")
        assert state.size == len('{"counter": 7}')

    def test_empty_prefix_returns_empty(self, ns: InMemoryNamespace) -> None:
        entries = list_dir(ns, "missing/")
        assert entries == []

    def test_render_dir_first_with_trailing_slash(self) -> None:
        entries = [
            DirEntry(name="artefacts", kind="directory"),
            DirEntry(name="state.json", kind="file", size=42),
        ]
        rendered = render_ls(entries)
        assert "artefacts/" in rendered
        assert "state.json" in rendered
        # File size shown
        assert "42B" in rendered

    def test_render_empty(self) -> None:
        assert render_ls([]) == "_(empty)_"


# --------------------------------------------------------------------------- #
# read_doc                                                                     #
# --------------------------------------------------------------------------- #

class TestReadDoc:
    def test_read_whole_doc(self, ns: InMemoryNamespace) -> None:
        content, meta = read_doc(ns, "demo/state.json")
        assert content == '{"counter": 7}'
        assert meta["truncated"] is False
        assert meta["total_lines"] == 1

    def test_offset_and_limit(self, ns: InMemoryNamespace) -> None:
        # event_log has 4 lines (counting blank lines from \n splits)
        content, meta = read_doc(ns, "demo/event_log.md", offset=3, limit=1)
        assert "add_item" in content
        assert meta["first_line"] == 3
        assert meta["last_line"] == 3
        # Truncation flag depends on whether more content remains
        assert meta["total_lines"] >= 3

    def test_missing_returns_none(self, ns: InMemoryNamespace) -> None:
        content, meta = read_doc(ns, "demo/missing.md")
        assert content is None
        assert meta == {}

    def test_offset_past_end(self, ns: InMemoryNamespace) -> None:
        content, meta = read_doc(ns, "demo/state.json", offset=999)
        assert content == ""
        assert meta["offset_out_of_range"] is True

    def test_render_missing(self) -> None:
        assert "no such document" in render_read(None, "demo/x", {})

    def test_render_truncated_emits_continuation_hint(self) -> None:
        rendered = render_read(
            "first chunk",
            "demo/event_log.md",
            {
                "first_line": 1,
                "last_line": 12,
                "total_lines": 50,
                "truncated": True,
            },
        )
        assert "offset=13" in rendered
        assert "of 50" in rendered

    def test_render_offset_out_of_range(self) -> None:
        rendered = render_read("", "demo/x", {"total_lines": 5, "offset_out_of_range": True})
        assert "offset past end" in rendered


# --------------------------------------------------------------------------- #
# grep_docs                                                                    #
# --------------------------------------------------------------------------- #

class TestGrep:
    def test_finds_matches_across_docs(self, ns: InMemoryNamespace) -> None:
        matches = grep_docs(ns, "Acme")
        paths = {m.path for m in matches}
        # "Acme" appears in two artefacts under demo/
        assert "demo/artefacts/legal_brief.md" in paths
        assert "demo/artefacts/notes.txt" in paths

    def test_path_prefix_filter(self, ns: InMemoryNamespace) -> None:
        matches = grep_docs(ns, "state", path="other/")
        # Restricted to other/, so demo/ matches are excluded
        for m in matches:
            assert m.path.startswith("other/")

    def test_glob_filter(self, ns: InMemoryNamespace) -> None:
        matches = grep_docs(ns, "Acme", glob="*.md")
        # Only .md files match the glob
        for m in matches:
            assert m.path.endswith(".md")

    def test_ignore_case(self, ns: InMemoryNamespace) -> None:
        case_sensitive = grep_docs(ns, "acme")
        case_insensitive = grep_docs(ns, "acme", ignore_case=True)
        assert len(case_insensitive) > len(case_sensitive)

    def test_limit_caps_results(self, ns: InMemoryNamespace) -> None:
        matches = grep_docs(ns, ".", limit=2)  # `.` matches every line
        assert len(matches) == 2

    def test_bad_regex_falls_back_to_literal(self, ns: InMemoryNamespace) -> None:
        # Unterminated group — would normally raise re.error. We fall back
        # to literal-substring semantics so the agent doesn't get a crash.
        matches = grep_docs(ns, "Acme(")
        # No doc literally contains "Acme(", so zero matches but no crash.
        assert matches == []

    def test_render_includes_path_lineno_and_line(self, ns: InMemoryNamespace) -> None:
        matches = grep_docs(ns, "Acme")
        rendered = render_grep(matches)
        assert "demo/artefacts/" in rendered
        assert ":" in rendered  # path:lineno separator

    def test_render_empty(self) -> None:
        assert render_grep([]) == "_(no matches)_"

    def test_render_limit_hint(self) -> None:
        matches = [
            __import__("distributed_agent_harness.namespace_browse", fromlist=["GrepMatch"]).GrepMatch(
                path="demo/x.md", line_number=1, line="match",
            )
            for _ in range(5)
        ]
        rendered = render_grep(matches, limit=5)
        assert "limit=5" in rendered
