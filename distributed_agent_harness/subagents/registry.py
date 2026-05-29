"""
AgentRegistry — discovery for subagents.

Two concrete implementations ship:

- ``StaticAgentRegistry`` — in-code list of ``AgentCard``s. Default for
  tests, demos, and small deployments where the roster is known.
- ``HttpAgentRegistry`` — talks to a corporate registry service over HTTP
  (the BFA pattern). Search filters are mapped to query params via an
  injectable ``query_param_mapping`` callable so different registry
  backends can be adapted without subclassing.

A helper, ``load_subagents_from_registry``, wraps a registry's search
results as ``A2ASubagent`` clients and registers them on a runtime in one
call.
"""
from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx

from .a2a import A2ASubagent, AuthHeaders
from .base import AgentCard, Skill

if TYPE_CHECKING:
    from ..runtime import AgentRuntime


# --------------------------------------------------------------------------- #
# Abstract registry                                                            #
# --------------------------------------------------------------------------- #

class AgentRegistry(ABC):
    """Pluggable directory of available subagents.

    Two concerns:
    - ``search`` returns cards matching client-side filters (the impl
      decides how to apply them).
    - ``get`` returns one card by name.
    """

    @abstractmethod
    async def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[AgentCard]: ...

    @abstractmethod
    async def get(self, name: str) -> AgentCard: ...


# --------------------------------------------------------------------------- #
# Static (in-memory) registry                                                  #
# --------------------------------------------------------------------------- #

class StaticAgentRegistry(AgentRegistry):
    """In-code list of cards. Filters in memory by walking the list."""

    def __init__(self, cards: list[AgentCard]) -> None:
        self._cards = list(cards)
        self._by_name = {c.name: c for c in self._cards}

    async def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[AgentCard]:
        results: list[AgentCard] = []
        for card in self._cards:
            if not _matches_query(card, query):
                continue
            if not _matches_tags(card, tags):
                continue
            results.append(card)
            if len(results) >= limit:
                break
        return results

    async def get(self, name: str) -> AgentCard:
        if name not in self._by_name:
            raise KeyError(f"No agent named {name!r} in StaticAgentRegistry")
        return self._by_name[name]


def _matches_query(card: AgentCard, query: str | None) -> bool:
    if not query:
        return True
    needle = query.lower()
    return needle in card.name.lower() or needle in card.description.lower()


def _matches_tags(card: AgentCard, tags: list[str] | None) -> bool:
    if not tags:
        return True
    card_tags: set[str] = set()
    for skill in card.skills:
        card_tags.update(skill.tags)
    return all(tag in card_tags for tag in tags)


# --------------------------------------------------------------------------- #
# HTTP registry                                                                #
# --------------------------------------------------------------------------- #

#: Callable that maps our standard filters into HTTP query params for a
#: specific corporate registry backend. Defaults to passing them through
#: 1:1 (with ``tags`` joined into a comma string).
QueryParamMapping = Callable[[dict[str, Any]], dict[str, str]]


def _default_query_mapping(filters: dict[str, Any]) -> dict[str, str]:
    params: dict[str, str] = {}
    if filters.get("query"):
        params["q"] = str(filters["query"])
    if filters.get("tags"):
        params["tags"] = ",".join(filters["tags"])
    if filters.get("limit") is not None:
        params["limit"] = str(filters["limit"])
    return params


class HttpAgentRegistry(AgentRegistry):
    """Registry backed by a corporate HTTP service (the BFA pattern).

    Decodes responses as A2A ``AgentCard`` JSON. The service must expose
    ``GET <base_url>/agents?<query params>`` returning ``{"agents": [card,
    ...]}`` and ``GET <base_url>/agents/{name}`` returning a single card.
    Adapt other backends via ``query_param_mapping`` rather than
    subclassing.
    """

    def __init__(
        self,
        base_url: str,
        auth: AuthHeaders = None,
        query_param_mapping: QueryParamMapping = _default_query_mapping,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._map = query_param_mapping
        self._client = client
        self._owns_client = client is None

    async def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[AgentCard]:
        params = self._map({"query": query, "tags": tags, "limit": limit})
        body = await self._get_json(f"{self._base_url}/agents", params=params)
        cards_json = body.get("agents", []) if isinstance(body, dict) else []
        return [_card_from_json(c) for c in cards_json if isinstance(c, dict)]

    async def get(self, name: str) -> AgentCard:
        body = await self._get_json(f"{self._base_url}/agents/{name}")
        if not isinstance(body, dict):
            raise ValueError(f"Registry returned non-object for agent {name!r}")
        return _card_from_json(body)

    # ----------------------------------------------------------------------- #
    # Internals                                                                #
    # ----------------------------------------------------------------------- #

    def _resolve_auth(self) -> dict[str, str]:
        if self._auth is None:
            return {}
        if callable(self._auth):
            return dict(self._auth())
        return dict(self._auth)

    async def _get_json(self, url: str, params: dict[str, str] | None = None) -> Any:
        client = self._client or httpx.AsyncClient()
        try:
            resp = await client.get(url, params=params, headers=self._resolve_auth())
            resp.raise_for_status()
            return resp.json()
        finally:
            if self._owns_client:
                await client.aclose()


def _card_from_json(data: dict[str, Any]) -> AgentCard:
    skills_json = data.get("skills", []) or []
    skills = [
        Skill(
            name=s.get("name", ""),
            description=s.get("description", ""),
            tags=list(s.get("tags") or []),
        )
        for s in skills_json
        if isinstance(s, dict)
    ]
    provider = data.get("provider")
    if isinstance(provider, dict):
        provider = provider.get("organization") or provider.get("name")
    return AgentCard(
        name=data.get("name", ""),
        description=data.get("description", ""),
        url=data.get("url", ""),
        skills=skills,
        provider=provider if isinstance(provider, str) else None,
        metadata={k: v for k, v in data.items() if k not in {
            "name", "description", "url", "skills", "provider",
        }},
    )


# --------------------------------------------------------------------------- #
# Bulk-load helper                                                             #
# --------------------------------------------------------------------------- #

async def load_subagents_from_registry(
    runtime: "AgentRuntime",
    registry: AgentRegistry,
    *,
    query: str | None = None,
    tags: list[str] | None = None,
    limit: int = 100,
    auth: AuthHeaders = None,
) -> list[A2ASubagent]:
    """Search ``registry``, wrap each result as ``A2ASubagent``, register
    them on ``runtime``, and return the list.

    ``auth`` is applied uniformly to every subagent created. For per-card
    auth, register subagents individually instead.
    """
    cards = await registry.search(query=query, tags=tags, limit=limit)
    subagents: list[A2ASubagent] = []
    for card in cards:
        sub = A2ASubagent(card=card, auth=auth)
        runtime.subagents.register(sub)
        subagents.append(sub)
    return subagents
