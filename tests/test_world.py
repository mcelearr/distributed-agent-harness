"""
Tests for BaseWorldEnvironment core lifecycle:
- @action wrapping (lock / hydrate / execute / flush / release)
- State persistence across instances
- Audit log correctness
- PromptBuilder output
"""
from __future__ import annotations

import json
import threading

import pytest
from pydantic import BaseModel

from distributed_agent_harness.adapters import InMemoryNamespace
from distributed_agent_harness.concurrency_handlers import InProcessLock
from distributed_agent_harness.prompt_builder import PromptBuilder
from distributed_agent_harness.world import BaseWorldEnvironment, action


# --------------------------------------------------------------------------- #
# Minimal world for testing                                                    #
# --------------------------------------------------------------------------- #

class CounterState(BaseModel):
    counter: int = 0
    items: list[str] = []


class CounterWorld(BaseWorldEnvironment):
    State = CounterState

    @action
    def increment(self, by: int = 1) -> int:
        """Increment the counter by *by*."""
        self.state.counter += by
        return self.state.counter

    @action
    def add_item(self, text: str) -> str:
        """Append *text* to the items list."""
        self.state.items.append(text)
        return text

    def not_an_action(self) -> str:
        """This method is NOT decorated with @action."""
        return "plain method"


@pytest.fixture
def namespace() -> InMemoryNamespace:
    return InMemoryNamespace()


@pytest.fixture
def world(namespace: InMemoryNamespace) -> CounterWorld:
    return CounterWorld(
        project_id="test",
        namespace=namespace,
        concurrency=InProcessLock(),
    )


# --------------------------------------------------------------------------- #
# @action lifecycle                                                            #
# --------------------------------------------------------------------------- #

class TestActionLifecycle:
    def test_action_mutates_state(self, world: CounterWorld) -> None:
        world.increment(5)
        assert world.state.counter == 5

    def test_multiple_actions_accumulate(self, world: CounterWorld) -> None:
        world.increment(3)
        world.increment(7)
        assert world.state.counter == 10

    def test_action_returns_value(self, world: CounterWorld) -> None:
        result = world.increment(4)
        assert result == 4

    def test_action_with_list(self, world: CounterWorld) -> None:
        world.add_item("hello")
        world.add_item("world")
        assert world.state.items == ["hello", "world"]


# --------------------------------------------------------------------------- #
# State persistence                                                            #
# --------------------------------------------------------------------------- #

class TestPersistence:
    def test_state_written_to_namespace(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.increment(99)
        raw = namespace.read_doc("test/state.json")
        assert raw is not None
        data = json.loads(raw)
        assert data["counter"] == 99

    def test_second_instance_sees_persisted_state(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        """A new instance pointing at the same namespace reads the saved state."""
        world.add_item("persisted")

        world2 = CounterWorld(
            project_id="test",
            namespace=namespace,
            concurrency=InProcessLock(),
        )
        assert "persisted" in world2.state.items

    def test_hydrate_on_init(self, namespace: InMemoryNamespace) -> None:
        """Constructor loads any existing state from the namespace."""
        # Manually pre-seed the namespace
        state = CounterState(counter=42)
        namespace.write_doc("test/state.json", state.model_dump_json())

        world = CounterWorld(
            project_id="test",
            namespace=namespace,
            concurrency=InProcessLock(),
        )
        assert world.state.counter == 42

    def test_action_always_reads_latest_state(
        self, namespace: InMemoryNamespace
    ) -> None:
        """Each @action re-reads the namespace before executing (never stale)."""
        world1 = CounterWorld("test", namespace, InProcessLock())
        world2 = CounterWorld("test", namespace, InProcessLock())

        world1.increment(10)
        # world2's in-memory state is stale (still 0), but the @action
        # wrapper calls _hydrate() first, so it sees counter=10 and adds 5
        world2.increment(5)

        assert world2.state.counter == 15


# --------------------------------------------------------------------------- #
# Audit log                                                                    #
# --------------------------------------------------------------------------- #

class TestAuditLog:
    def test_audit_entry_written(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.increment(1)
        raw = namespace.read_doc("test/audit.jsonl")
        assert raw is not None
        entry = json.loads(raw.strip().split("\n")[0])
        assert entry["method"] == "increment"
        assert entry["project_id"] == "test"

    def test_audit_is_append_only(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.increment(1)
        world.add_item("x")
        world.increment(2)
        raw = namespace.read_doc("test/audit.jsonl")
        assert raw is not None
        lines = [l for l in raw.strip().split("\n") if l]
        assert len(lines) == 3

    def test_audit_records_error(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        with pytest.raises(TypeError):
            world.increment("not_an_int")  # type: ignore

        raw = namespace.read_doc("test/audit.jsonl")
        assert raw is not None
        entry = json.loads(raw.strip().split("\n")[0])
        assert "error" in entry

    def test_audit_has_timestamp(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.increment()
        raw = namespace.read_doc("test/audit.jsonl")
        assert raw is not None
        entry = json.loads(raw.strip())
        assert "timestamp" in entry
        assert "T" in entry["timestamp"]  # ISO 8601


# --------------------------------------------------------------------------- #
# get_actions introspection                                                    #
# --------------------------------------------------------------------------- #

class TestGetActions:
    def test_returns_decorated_methods(self) -> None:
        actions = CounterWorld.get_actions()
        assert "increment" in actions
        assert "add_item" in actions

    def test_excludes_plain_methods(self) -> None:
        actions = CounterWorld.get_actions()
        assert "not_an_action" not in actions

    def test_excludes_private_methods(self) -> None:
        actions = CounterWorld.get_actions()
        private = [k for k in actions if k.startswith("_")]
        assert private == []

    def test_action_has_source(self) -> None:
        actions = CounterWorld.get_actions()
        assert actions["increment"]._source != ""
        assert "def increment" in actions["increment"]._source


# --------------------------------------------------------------------------- #
# PromptBuilder                                                                #
# --------------------------------------------------------------------------- #

class TestPromptBuilder:
    def test_builds_actions_prompt(self) -> None:
        builder = PromptBuilder(CounterWorld)
        prompt = builder.build_actions_prompt()
        assert "## Available Actions" in prompt
        assert "increment" in prompt
        assert "add_item" in prompt

    def test_includes_docstrings(self) -> None:
        builder = PromptBuilder(CounterWorld)
        prompt = builder.build_actions_prompt()
        assert "Increment the counter" in prompt

    def test_includes_source_when_requested(self) -> None:
        builder = PromptBuilder(CounterWorld, include_source=True)
        prompt = builder.build_actions_prompt()
        assert "```python" in prompt

    def test_state_prompt(self, world: CounterWorld) -> None:
        world.increment(7)
        builder = PromptBuilder(CounterWorld)
        prompt = builder.build_state_prompt(world)
        assert "## Current State" in prompt
        assert "7" in prompt

    def test_full_prompt_combines_sections(self, world: CounterWorld) -> None:
        builder = PromptBuilder(CounterWorld)
        prompt = builder.build_full_prompt(world)
        # The standardised sections must all be present, in order.
        for section in [
            "## Project Summary",
            "## Available Actions",
            "## Recent Activity",
            "## Current State",
        ]:
            assert section in prompt, f"missing section: {section}"


# --------------------------------------------------------------------------- #
# Standardised namespace documents — summary.md and event_log.md               #
# --------------------------------------------------------------------------- #

class TestSummaryDoc:
    def test_summary_written_after_action(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.increment()
        summary = namespace.read_doc("test/summary.md")
        assert summary is not None
        # Default summary mentions the class name and the project id
        assert "CounterWorld" in summary
        assert "test" in summary

    def test_summary_updates_on_each_action(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.increment(1)
        first = namespace.read_doc("test/summary.md")
        world.increment(99)
        second = namespace.read_doc("test/summary.md")
        # The default summary embeds the state JSON, so it should change
        assert first != second
        assert "100" in second  # 1 + 99

    def test_subclass_override_takes_effect(
        self, namespace: InMemoryNamespace
    ) -> None:
        class FancyState(BaseModel):
            x: int = 0

        class FancyWorld(BaseWorldEnvironment):
            State = FancyState

            def render_summary(self) -> str:
                return f"# Fancy\n\nx = {self.state.x}\n"

            @action
            def bump(self) -> int:
                self.state.x += 1
                return self.state.x

        world = FancyWorld("p1", namespace, InProcessLock())
        world.bump()
        summary = namespace.read_doc("p1/summary.md")
        assert summary == "# Fancy\n\nx = 1\n"

    def test_buggy_render_summary_does_not_break_state_write(
        self, namespace: InMemoryNamespace
    ) -> None:
        """A subclass that raises in render_summary() should still persist state."""

        class S(BaseModel):
            v: int = 0

        class BuggyWorld(BaseWorldEnvironment):
            State = S

            def render_summary(self) -> str:
                raise RuntimeError("oops")

            @action
            def set_v(self, value: int) -> int:
                self.state.v = value
                return value

        world = BuggyWorld("p1", namespace, InProcessLock())
        world.set_v(42)
        # State.json was still written
        assert "42" in namespace.read_doc("p1/state.json")
        # Summary fell back to the default
        summary = namespace.read_doc("p1/summary.md")
        assert summary is not None
        assert "No `render_summary()` override" in summary


class TestEventLogDoc:
    def test_event_log_created_on_first_action(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.increment()
        log = namespace.read_doc("test/event_log.md")
        assert log is not None
        assert log.startswith("# Event Log")
        assert "increment" in log

    def test_event_log_appends_one_line_per_action(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.increment(1)
        world.add_item("a")
        world.increment(2)

        log = namespace.read_doc("test/event_log.md")
        bullets = [line for line in log.split("\n") if line.startswith("- ")]
        assert len(bullets) == 3
        assert "increment" in bullets[0]
        assert "add_item" in bullets[1]
        assert "increment" in bullets[2]

    def test_event_log_records_failures(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        with pytest.raises(TypeError):
            world.increment("bad")  # type: ignore[arg-type]
        log = namespace.read_doc("test/event_log.md")
        assert "FAILED" in log

    def test_event_log_truncates_long_args(
        self, world: CounterWorld, namespace: InMemoryNamespace
    ) -> None:
        world.add_item("x" * 500)
        log = namespace.read_doc("test/event_log.md")
        # No single line should be obscenely long
        for line in log.split("\n"):
            assert len(line) < 200, f"line too long: {line[:80]}..."


class TestRecentActivityInPrompt:
    def test_prompt_includes_recent_event_log_lines(
        self, world: CounterWorld
    ) -> None:
        world.increment()
        world.add_item("hello")
        builder = PromptBuilder(CounterWorld)
        prompt = builder.build_recent_activity_prompt(world)
        assert "increment" in prompt
        assert "add_item" in prompt

    def test_prompt_truncates_to_last_n(self, world: CounterWorld) -> None:
        for i in range(20):
            world.increment()
        builder = PromptBuilder(CounterWorld, recent_activity_count=5)
        prompt = builder.build_recent_activity_prompt(world)
        bullets = [line for line in prompt.split("\n") if line.startswith("- ")]
        assert len(bullets) == 5

    def test_prompt_when_no_activity_yet(self, world: CounterWorld) -> None:
        builder = PromptBuilder(CounterWorld)
        prompt = builder.build_recent_activity_prompt(world)
        assert "no actions taken" in prompt


# --------------------------------------------------------------------------- #
# Concurrency                                                                  #
# --------------------------------------------------------------------------- #

class TestConcurrency:
    def test_concurrent_increments_are_safe(
        self, namespace: InMemoryNamespace
    ) -> None:
        """10 threads each incrementing 10 times should give counter == 100."""
        concurrency = InProcessLock()
        world = CounterWorld("test", namespace, concurrency)

        def run() -> None:
            for _ in range(10):
                world.increment(1)

        threads = [threading.Thread(target=run) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert world.state.counter == 100

    def test_missing_state_class_raises(self, namespace: InMemoryNamespace) -> None:
        class BadWorld(BaseWorldEnvironment):
            pass  # No State defined

        with pytest.raises(TypeError, match="State"):
            BadWorld("test", namespace, InProcessLock())
