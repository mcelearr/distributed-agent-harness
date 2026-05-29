"""
Tests for the subagent layer:

- A2ASubagent against an in-memory SSE server using ``httpx.MockTransport``
- SubagentRegistry registration / lookup
- StaticAgentRegistry filtering by query / tags
- HttpAgentRegistry against a mocked corporate registry HTTP service
- ``load_subagents_from_registry`` end-to-end
- Auth resolution (dict and callable forms)
- SubagentTimeout when the SSE stream stalls
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest

from distributed_agent_harness.subagents import (
    A2ASubagent,
    AgentCard,
    HttpAgentRegistry,
    Skill,
    StaticAgentRegistry,
    SubagentRegistry,
    SubagentTimeout,
)


# --------------------------------------------------------------------------- #
# Helpers: build an httpx MockTransport that speaks our subset of A2A         #
# --------------------------------------------------------------------------- #

def _sse_lines(events: list[tuple[str, dict | str]]) -> str:
    """Serialise ``(event_type, payload)`` pairs as an SSE response body."""
    out: list[str] = []
    for event_type, payload in events:
        out.append(f"event: {event_type}")
        if isinstance(payload, dict):
            out.append(f"data: {json.dumps(payload)}")
        else:
            out.append(f"data: {payload}")
        out.append("")  # blank line dispatches the event
    return "\n".join(out) + "\n"


def _make_transport(
    *,
    submit_response: dict | None = None,
    sse_body: str = "",
    on_submit: Callable[[httpx.Request], None] | None = None,
) -> httpx.MockTransport:
    """A MockTransport that routes /tasks/send → JSON, /tasks/<id>/events → SSE."""
    submit_response = submit_response or {
        "taskId": "task-1",
        "contextId": "ctx-1",
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tasks/send"):
            if on_submit is not None:
                on_submit(request)
            return httpx.Response(200, json=submit_response)
        if "/tasks/" in request.url.path and request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                content=sse_body.encode("utf-8"),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    return httpx.MockTransport(_handler)


def _make_card(name: str = "test_agent") -> AgentCard:
    return AgentCard(
        name=name,
        description=f"A test {name}",
        url="https://example.test",
        skills=[
            Skill(name="case-search", description="Search cases", tags=["legal", "uk"]),
        ],
        provider="Acme",
    )


# --------------------------------------------------------------------------- #
# A2ASubagent                                                                  #
# --------------------------------------------------------------------------- #

class TestA2AConsult:
    @pytest.mark.asyncio
    async def test_completed_response_returned(self) -> None:
        sse = _sse_lines([
            ("working", {"text": "thinking… "}),
            ("working", {"text": "almost done… "}),
            ("completed", {
                "text": "Under UK GDPR Art. 33, notify within 72 hours.",
                "contextId": "ctx-1",
            }),
        ])
        transport = _make_transport(sse_body=sse)
        client = httpx.AsyncClient(transport=transport)
        agent = A2ASubagent(_make_card(), client=client)

        progress: list[str] = []
        async def on_progress(delta: str) -> None:
            progress.append(delta)

        response = await agent.consult(
            message="What is Art. 33?", on_progress=on_progress,
        )
        await client.aclose()

        assert response.status == "completed"
        assert "72 hours" in response.content
        assert response.session_id == "ctx-1"
        assert progress == ["thinking… ", "almost done… "]

    @pytest.mark.asyncio
    async def test_input_required_response_returned(self) -> None:
        sse = _sse_lines([
            ("input-required", {
                "text": "Which jurisdiction — UK or EU?",
                "contextId": "ctx-2",
            }),
        ])
        transport = _make_transport(sse_body=sse)
        client = httpx.AsyncClient(transport=transport)
        agent = A2ASubagent(_make_card(), client=client)

        response = await agent.consult(message="Help me with GDPR")
        await client.aclose()

        assert response.status == "input-required"
        assert "jurisdiction" in response.content.lower()
        assert response.session_id == "ctx-2"

    @pytest.mark.asyncio
    async def test_session_id_round_trips_into_submit(self) -> None:
        sse = _sse_lines([("completed", {"text": "OK", "contextId": "ctx-3"})])
        captured: list[httpx.Request] = []
        transport = _make_transport(
            sse_body=sse,
            on_submit=lambda req: captured.append(req),
        )
        client = httpx.AsyncClient(transport=transport)
        agent = A2ASubagent(_make_card(), client=client)

        await agent.consult(message="follow up", session_id="ctx-3")
        await client.aclose()

        body = json.loads(captured[0].content.decode("utf-8"))
        assert body["contextId"] == "ctx-3"
        assert body["message"]["parts"][0]["text"] == "follow up"

    @pytest.mark.asyncio
    async def test_keep_alive_comments_do_not_trigger_timeout(self) -> None:
        # Keep-alive comments mixed in — watchdog should be reset by each.
        sse = (
            ": ping\n\n"
            ": ping\n\n"
            "event: completed\n"
            'data: {"text": "done", "contextId": "ctx-1"}\n\n'
        )
        transport = _make_transport(sse_body=sse)
        client = httpx.AsyncClient(transport=transport)
        agent = A2ASubagent(_make_card(), client=client)

        response = await agent.consult(message="ping me", timeout=5.0)
        await client.aclose()

        assert response.status == "completed"
        assert response.content == "done"


class TestA2AAuth:
    @pytest.mark.asyncio
    async def test_static_dict_auth_applied_on_submit(self) -> None:
        captured: list[httpx.Request] = []
        transport = _make_transport(
            sse_body=_sse_lines([("completed", {"text": "ok"})]),
            on_submit=lambda req: captured.append(req),
        )
        client = httpx.AsyncClient(transport=transport)
        agent = A2ASubagent(
            _make_card(),
            auth={"Authorization": "Bearer static-abc"},
            client=client,
        )

        await agent.consult(message="hello")
        await client.aclose()

        assert captured[0].headers["Authorization"] == "Bearer static-abc"

    @pytest.mark.asyncio
    async def test_callable_auth_recomputed_each_call(self) -> None:
        call_count = 0
        def fresh_headers() -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"Authorization": f"Bearer dynamic-{call_count}"}

        captured: list[httpx.Request] = []
        transport = _make_transport(
            sse_body=_sse_lines([("completed", {"text": "ok"})]),
            on_submit=lambda req: captured.append(req),
        )
        client = httpx.AsyncClient(transport=transport)
        agent = A2ASubagent(_make_card(), auth=fresh_headers, client=client)

        await agent.consult(message="first")
        await agent.consult(message="second")
        await client.aclose()

        assert captured[0].headers["Authorization"] == "Bearer dynamic-1"
        assert captured[1].headers["Authorization"] == "Bearer dynamic-2"


# --------------------------------------------------------------------------- #
# StaticAgentRegistry                                                          #
# --------------------------------------------------------------------------- #

class TestStaticAgentRegistry:
    @pytest.mark.asyncio
    async def test_no_filters_returns_all(self) -> None:
        registry = StaticAgentRegistry([_make_card("a"), _make_card("b")])
        results = await registry.search()
        assert {c.name for c in results} == {"a", "b"}

    @pytest.mark.asyncio
    async def test_query_matches_name_and_description(self) -> None:
        registry = StaticAgentRegistry([
            AgentCard(name="legal_research", description="case search", url=""),
            AgentCard(name="doc_drafting", description="write letters", url=""),
        ])
        # Match by name
        assert {c.name for c in await registry.search(query="legal")} == {"legal_research"}
        # Match by description
        assert {c.name for c in await registry.search(query="letters")} == {"doc_drafting"}

    @pytest.mark.asyncio
    async def test_tag_filter_uses_skill_tags(self) -> None:
        legal = AgentCard(
            name="legal_research", description="", url="",
            skills=[Skill(name="search", tags=["legal", "uk"])],
        )
        doc = AgentCard(
            name="doc_drafting", description="", url="",
            skills=[Skill(name="draft", tags=["docs"])],
        )
        registry = StaticAgentRegistry([legal, doc])
        results = await registry.search(tags=["legal"])
        assert {c.name for c in results} == {"legal_research"}

    @pytest.mark.asyncio
    async def test_get_raises_keyerror_for_unknown(self) -> None:
        registry = StaticAgentRegistry([])
        with pytest.raises(KeyError):
            await registry.get("nope")


# --------------------------------------------------------------------------- #
# SubagentRegistry                                                             #
# --------------------------------------------------------------------------- #

class TestSubagentRegistry:
    def test_register_and_lookup(self) -> None:
        registry = SubagentRegistry()
        agent = A2ASubagent(_make_card("legal"))
        registry.register(agent)
        assert registry.get("legal") is agent
        assert "legal" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self) -> None:
        registry = SubagentRegistry()
        registry.register(A2ASubagent(_make_card("dup")))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(A2ASubagent(_make_card("dup")))

    def test_unregister(self) -> None:
        registry = SubagentRegistry()
        registry.register(A2ASubagent(_make_card("legal")))
        registry.unregister("legal")
        assert registry.get("legal") is None
        # Unregistering an absent name is a no-op
        registry.unregister("legal")


# --------------------------------------------------------------------------- #
# Timeout                                                                      #
# --------------------------------------------------------------------------- #

class TestSubagentTimeout:
    @pytest.mark.asyncio
    async def test_stalled_stream_raises_timeout(self) -> None:
        # Build a transport whose SSE stream never produces lines.
        async def _stalled_stream():
            # Sleep longer than the timeout, then yield nothing.
            await asyncio.sleep(10)
            yield b""

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/tasks/send"):
                return httpx.Response(200, json={"taskId": "t", "contextId": "c"})
            if request.url.path.endswith("/events"):
                # An empty streaming body — httpx will return EOF, which is not
                # a stall. To force a stall we'd need a transport that blocks.
                # Use an indefinitely-pending content stream instead.
                async def _never():
                    await asyncio.sleep(10)
                    yield b""  # pragma: no cover
                return httpx.Response(
                    200,
                    content=_never(),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport)
        agent = A2ASubagent(_make_card(), client=client)

        with pytest.raises(SubagentTimeout):
            await agent.consult(message="hi", timeout=0.2)
        await client.aclose()


# --------------------------------------------------------------------------- #
# HttpAgentRegistry                                                            #
# --------------------------------------------------------------------------- #

def _registry_transport(
    *,
    list_response: dict | list | None = None,
    get_responses: dict[str, dict] | None = None,
    capture: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """A MockTransport that emulates a tiny corporate agent registry service.

    Routes:
    - ``GET /agents`` → JSON ``list_response`` (defaults to empty list)
    - ``GET /agents/{name}`` → JSON ``get_responses[name]``
    """
    list_response = list_response if list_response is not None else {"agents": []}
    get_responses = get_responses or {}

    def _handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        path = request.url.path
        if path.endswith("/agents"):
            return httpx.Response(200, json=list_response)
        if "/agents/" in path:
            name = path.rsplit("/", 1)[-1]
            if name in get_responses:
                return httpx.Response(200, json=get_responses[name])
            return httpx.Response(404)
        return httpx.Response(404)

    return httpx.MockTransport(_handler)


def _registry_card_json(name: str, **overrides: Any) -> dict:
    """An A2A AgentCard JSON shape with sensible defaults."""
    base: dict[str, Any] = {
        "name": name,
        "description": f"A test {name} agent",
        "url": f"https://internal/{name}",
        "skills": [
            {"name": "case-search", "description": "Search cases", "tags": ["legal", "uk"]},
        ],
        "provider": {"organization": "Acme Legal"},
    }
    base.update(overrides)
    return base


class TestHttpAgentRegistrySearch:
    @pytest.mark.asyncio
    async def test_search_decodes_cards(self) -> None:
        transport = _registry_transport(list_response={
            "agents": [_registry_card_json("legal_research")],
        })
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(base_url="https://registry", client=client)

        results = await registry.search()
        await client.aclose()

        assert len(results) == 1
        assert results[0].name == "legal_research"
        assert results[0].url == "https://internal/legal_research"
        # provider object → flat string
        assert results[0].provider == "Acme Legal"
        # skills carry their tags
        assert results[0].skills[0].tags == ["legal", "uk"]

    @pytest.mark.asyncio
    async def test_search_passes_filters_as_query_params(self) -> None:
        captured: list[httpx.Request] = []
        transport = _registry_transport(
            list_response={"agents": []},
            capture=captured,
        )
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(base_url="https://registry", client=client)

        await registry.search(query="legal", tags=["uk", "litigation"], limit=50)
        await client.aclose()

        sent_url = captured[-1].url
        assert sent_url.params.get("q") == "legal"
        # default mapping joins tags with comma
        assert sent_url.params.get("tags") == "uk,litigation"
        assert sent_url.params.get("limit") == "50"

    @pytest.mark.asyncio
    async def test_custom_query_param_mapping_is_used(self) -> None:
        captured: list[httpx.Request] = []
        transport = _registry_transport(
            list_response={"agents": []},
            capture=captured,
        )
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")

        def custom_mapping(filters: dict[str, Any]) -> dict[str, str]:
            # Different backend uses different param names.
            out: dict[str, str] = {}
            if filters.get("query"):
                out["search"] = str(filters["query"])
            if filters.get("tags"):
                out["category"] = filters["tags"][0]
            return out

        registry = HttpAgentRegistry(
            base_url="https://registry",
            client=client,
            query_param_mapping=custom_mapping,
        )
        await registry.search(query="acme", tags=["legal"])
        await client.aclose()

        sent_url = captured[-1].url
        assert sent_url.params.get("search") == "acme"
        assert sent_url.params.get("category") == "legal"
        # default-mapping params are NOT present
        assert sent_url.params.get("q") is None
        assert sent_url.params.get("tags") is None

    @pytest.mark.asyncio
    async def test_search_attaches_static_auth_headers(self) -> None:
        captured: list[httpx.Request] = []
        transport = _registry_transport(
            list_response={"agents": []},
            capture=captured,
        )
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(
            base_url="https://registry",
            client=client,
            auth={"Authorization": "Bearer reg-token"},
        )

        await registry.search()
        await client.aclose()

        assert captured[-1].headers["Authorization"] == "Bearer reg-token"

    @pytest.mark.asyncio
    async def test_search_callable_auth_is_recomputed(self) -> None:
        counter = 0

        def fresh() -> dict[str, str]:
            nonlocal counter
            counter += 1
            return {"Authorization": f"Bearer t-{counter}"}

        captured: list[httpx.Request] = []
        transport = _registry_transport(
            list_response={"agents": []},
            capture=captured,
        )
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(
            base_url="https://registry",
            client=client,
            auth=fresh,
        )

        await registry.search()
        await registry.search()
        await client.aclose()

        assert captured[0].headers["Authorization"] == "Bearer t-1"
        assert captured[1].headers["Authorization"] == "Bearer t-2"

    @pytest.mark.asyncio
    async def test_search_ignores_non_dict_entries(self) -> None:
        # Defensive: a registry that returns a malformed entry shouldn't
        # crash the harness; we just drop the bad row.
        transport = _registry_transport(list_response={
            "agents": [
                _registry_card_json("ok"),
                "not-an-object",
                {"name": "also_ok", "description": "", "url": ""},
            ],
        })
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(base_url="https://registry", client=client)

        results = await registry.search()
        await client.aclose()

        assert {c.name for c in results} == {"ok", "also_ok"}


class TestHttpAgentRegistryGet:
    @pytest.mark.asyncio
    async def test_get_returns_single_card(self) -> None:
        transport = _registry_transport(get_responses={
            "legal_research": _registry_card_json("legal_research"),
        })
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(base_url="https://registry", client=client)

        card = await registry.get("legal_research")
        await client.aclose()

        assert card.name == "legal_research"
        assert card.provider == "Acme Legal"

    @pytest.mark.asyncio
    async def test_get_missing_card_raises_http_error(self) -> None:
        transport = _registry_transport(get_responses={})
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(base_url="https://registry", client=client)

        with pytest.raises(httpx.HTTPStatusError):
            await registry.get("missing")
        await client.aclose()

    @pytest.mark.asyncio
    async def test_provider_can_be_plain_string(self) -> None:
        # Some registries return `provider` as a flat string. We tolerate.
        transport = _registry_transport(get_responses={
            "x": _registry_card_json("x", provider="Acme Inc."),
        })
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(base_url="https://registry", client=client)

        card = await registry.get("x")
        await client.aclose()

        assert card.provider == "Acme Inc."

    @pytest.mark.asyncio
    async def test_extra_fields_kept_in_metadata(self) -> None:
        transport = _registry_transport(get_responses={
            "x": _registry_card_json("x", version="1.2.3", cost_per_call=0.05),
        })
        client = httpx.AsyncClient(transport=transport, base_url="https://registry")
        registry = HttpAgentRegistry(base_url="https://registry", client=client)

        card = await registry.get("x")
        await client.aclose()

        assert card.metadata["version"] == "1.2.3"
        assert card.metadata["cost_per_call"] == 0.05
