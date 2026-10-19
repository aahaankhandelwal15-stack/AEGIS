"""Kairo — Memory Recall agent for the Memory Pod.

Handles recall queries, context bundling with token budgeting,
working memory synchronization, and thread fetching.
Also handles storage of goals, tasks, and execution logs.
"""

from __future__ import annotations

import json
import os
import sys
from enum import Enum
from typing import Any

# Add the project root to the sys.path so that the aegis package can be found
# when this script is run from within the aegis/agents/kairo directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tiktoken import get_encoding

from aegis.config import Settings, get_settings
from aegis.core.models import (
    AgentEnvelope,
    AgentResult,
    ClassificationResult,
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    Goal,
    Task,
    ExecutionLog,
)
from aegis.core.protocol import KAIRO_AGENT_ID, MEMORY_POD_ID
from aegis.memory_pod.store import MemoryStore


# Token encoding for estimating token counts (using cl100k_base for GPT-4)
_ENCODING = get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Return the number of tokens in a text string."""
    return len(_ENCODING.encode(text))


class KairoIntent(str, Enum):
    """Kairo-specific intents."""

    RECALL = "recall"
    SYNC_WORKING_MEMORY = "sync_working_memory"
    FETCH_THREAD = "fetch_thread"
    HEALTH = "health"
    STORE_GOAL = "store_goal"
    STORE_TASK = "store_task"
    STORE_EXECUTION_LOG = "store_execution_log"


class KairoAgent:
    """Kairo agent: read path for memory recall and context synching."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or MemoryStore(self._settings)

    @property
    def agent_id(self) -> str:
        return KAIRO_AGENT_ID

    @property
    def pod_id(self) -> str:
        return MEMORY_POD_ID

    def handle(self, envelope: AgentEnvelope) -> AgentResult:
        try:
            if envelope.agent_id != KAIRO_AGENT_ID or envelope.pod_id != MEMORY_POD_ID:
                return self._fail(
                    envelope,
                    f"Misrouted envelope: expected {MEMORY_POD_ID}/{KAIRO_AGENT_ID}",
                )

            intent = envelope.intent
            if intent == KairoIntent.HEALTH:
                return self._ok(
                    envelope,
                    {
                        "status": "healthy",
                        "memory_count": self._store.count(),
                    },
                )
            if intent == KairoIntent.RECALL:
                return self._recall(envelope)
            if intent == KairoIntent.SYNC_WORKING_MEMORY:
                return self._sync_working_memory(envelope)
            if intent == KairoIntent.FETCH_THREAD:
                return self._fetch_thread(envelope)
            if intent == KairoIntent.STORE_GOAL:
                return self._store_goal(envelope)
            if intent == KairoIntent.STORE_TASK:
                return self._store_task(envelope)
            if intent == KairoIntent.STORE_EXECUTION_LOG:
                return self._store_execution_log(envelope)
            return self._fail(envelope, f"Unsupported intent: {intent}")
        except Exception as exc:  # noqa: BLE001
            return self._fail(envelope, str(exc))

    # -----------------------------------------------------------------
    # Intent handlers
    # -----------------------------------------------------------------
    def _recall(self, envelope: AgentEnvelope) -> AgentResult:
        """Handle a recall request with token budgeting."""
        payload = envelope.payload
        query_text = payload.get("query")
        if not query_text or not str(query_text).strip():
            return self._fail(envelope, "payload.query is required for recall")

        # Extract parameters
        limit = int(payload.get("limit", 5))
        memory_types = payload.get("memory_types")
        session_id = payload.get("session_id")
        entity_id = payload.get("entity_id")
        requesting_agent_id = payload.get("requesting_agent_id", "unknown")
        token_budget = int(payload.get("token_budget", 512))  # default token budget

        # Build MemoryQuery
        query = MemoryQuery(
            query=str(query_text),
            limit=limit * 3,  # fetch more to allow for truncation
            memory_types=memory_types,
            session_id=session_id,
            entity_id=entity_id,
            requesting_agent_id=requesting_agent_id,
        )

        # Query the store
        hits = self._store.query(query)

        # Convert to list of dicts with token count
        candidates = []
        for hit in hits:
            record = hit.record
            text = record.content
            tokens = _count_tokens(text)
            candidates.append(
                {
                    "record": record,
                    "score": hit.score,
                    "text": text,
                    "tokens": tokens,
                }
            )

        # Sort by score descending (higher score = more relevant)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Apply token budget: select until we run out of budget
        selected = []
        total_tokens = 0
        for cand in candidates:
            if total_tokens + cand["tokens"] > token_budget:
                break
            selected.append(cand)
            total_tokens += cand["tokens"]

        # Format the result
        hits_out = []
        for cand in selected:
            hits_out.append(
                {
                    "score": cand["score"],
                    "memory": cand["record"].model_dump(mode="json"),
                }
            )

        return self._ok(
            envelope,
            {
                "hits": hits_out,
                "total_tokens": total_tokens,
                "token_budget": token_budget,
                "truncated": len(selected) < len(candidates),
            },
        )

    def _store_goal(self, envelope: AgentEnvelope) -> AgentResult:
        """Store a goal."""
        payload = envelope.payload
        # Expect payload to contain a goal dict that conforms to the Goal model
        try:
            goal = Goal(**payload)
        except Exception as e:
            return self._fail(envelope, f"Invalid goal data: {e}")

        # Store as a memory record
        record = MemoryRecord(
            project_id=self._settings.aegis_project_id,
            content=goal.model_dump_json(),  # Store as JSON string
            memory_type=MemoryType.GOAL,
            scope=MemoryScope.POD_SHARED,
            source_agent_id=envelope.agent_id or "unknown",
            session_id=payload.get("session_id", "unknown"),
            metadata=payload.get("metadata", {}),
        )
        stored_record = self._store.add(record)
        return self._ok(
            envelope,
            {
                "stored_goal": stored_record.model_dump(mode="json"),
            },
        )

    def _store_task(self, envelope: AgentEnvelope) -> AgentResult:
        """Store a task."""
        payload = envelope.payload
        try:
            task = Task(**payload)
        except Exception as e:
            return self._fail(envelope, f"Invalid task data: {e}")

        record = MemoryRecord(
            project_id=self._settings.aegis_project_id,
            content=task.model_dump_json(),
            memory_type=MemoryType.TASK,
            scope=MemoryScope.POD_SHARED,
            source_agent_id=envelope.agent_id or "unknown",
            session_id=payload.get("session_id", "unknown"),
            metadata=payload.get("metadata", {}),
        )
        stored_record = self._store.add(record)
        return self._ok(
            envelope,
            {
                "stored_task": stored_record.model_dump(mode="json"),
            },
        )

    def _store_execution_log(self, envelope: AgentEnvelope) -> AgentResult:
        """Store an execution log."""
        payload = envelope.payload
        try:
            execution_log = ExecutionLog(**payload)
        except Exception as e:
            return self._fail(envelope, f"Invalid execution log data: {e}")

        record = MemoryRecord(
            project_id=self._settings.aegis_project_id,
            content=execution_log.model_dump_json(),
            memory_type=MemoryType.EXECUTION,
            scope=MemoryScope.POD_SHARED,
            source_agent_id=envelope.agent_id or "unknown",
            session_id=payload.get("session_id", "unknown"),
            metadata=payload.get("metadata", {}),
        )
        stored_record = self._store.add(record)
        return self._ok(
            envelope,
            {
                "stored_execution_log": stored_record.model_dump(mode="json"),
            },
        )

    def _sync_working_memory(self, envelope: AgentEnvelope) -> AgentResult:
        """Stub for synchronizing an agent's working memory with long-term store."""
        # In a full implementation, we would:
        # 1. Load the agent's current working memory snapshot
        # 2. Query the long-term store for newer memories since the snapshot.
        # 3. Update the working memory with the new memories.
        # 4. Save the updated working memory.
        # For now, we return a placeholder.
        return self._ok(
            envelope,
            {
                "status": "synced",
                "message": "Working memory synchronization is not yet implemented.",
            },
        )

    def _fetch_thread(self, envelope: AgentEnvelope) -> AgentResult:
        """Stub for fetching a conversation/task thread when a session resumes."""
        # In a full implementation, we would:
        # 1. Use the session_id to find all memories in that session.
        # 2. Order them by timestamp to reconstruct the thread.
        # 3. Return the thread as a list of memories.
        # For now, we return a placeholder.
        return self._ok(
            envelope,
            {
                "status": "fetched",
                "message": "Thread fetching is not yet implemented.",
            },
        )

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------
    def _ok(self, envelope: AgentEnvelope, data: dict[str, Any]) -> AgentResult:
        return AgentResult(
            ok=True,
            correlation_id=envelope.correlation_id,
            agent_id=KAIRO_AGENT_ID,
            pod_id=MEMORY_POD_ID,
            data=data,
        )

    def _fail(self, envelope: AgentEnvelope, message: str) -> AgentResult:
        return AgentResult(
            ok=False,
            correlation_id=envelope.correlation_id,
            agent_id=KAIRO_AGENT_ID,
            pod_id=MEMORY_POD_ID,
            error=message,
        )