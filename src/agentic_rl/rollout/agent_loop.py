from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, Sequence

from agentic_rl.outcome.parser import parse_model_trajectory
from agentic_rl.retriever.protocol import RetrievalDocument


class AsyncRolloutReplica(Protocol):
    async def generate_action(
        self,
        conversation: Sequence[dict[str, str]],
        *,
        snapshot_step: int,
    ) -> str:
        ...


class AsyncRetriever(Protocol):
    async def retrieve_one(self, query: str) -> Sequence[RetrievalDocument]:
        ...


@dataclass(frozen=True)
class AgentLoopResult:
    conversation: tuple[dict[str, str], ...]
    model_actions: tuple[str, ...]
    search_queries: tuple[str, ...]
    termination_reason: str
    snapshot_step: int


class AgentLoop:
    def __init__(
        self,
        rollout: AsyncRolloutReplica,
        retriever: AsyncRetriever,
        *,
        max_search_turns: int,
        information_formatter: Callable[[Sequence[RetrievalDocument]], str],
    ) -> None:
        self.rollout = rollout
        self.retriever = retriever
        self.max_search_turns = int(max_search_turns)
        self.information_formatter = information_formatter

    async def run(
        self,
        initial_conversation: Sequence[dict[str, str]],
        *,
        snapshot_step: int,
    ) -> AgentLoopResult:
        conversation = [dict(message) for message in initial_conversation]
        actions: list[str] = []
        queries: list[str] = []
        termination = "max_search_turns"
        for _ in range(self.max_search_turns + 1):
            action = await self.rollout.generate_action(
                conversation,
                snapshot_step=snapshot_step,
            )
            actions.append(action)
            conversation.append({"role": "assistant", "content": action})
            parsed = parse_model_trajectory(actions)
            if parsed.valid:
                termination = "answer"
                break
            if parsed.parser_error_type == "missing_final_answer" and parsed.search_queries:
                latest_query = parsed.search_queries[-1]
                if len(parsed.search_queries) > self.max_search_turns:
                    termination = "max_search_turns"
                    break
                documents = await self.retriever.retrieve_one(latest_query)
                information = self.information_formatter(documents)
                conversation.append(
                    {
                        "role": "tool",
                        "content": f"<information>{information}</information>",
                    }
                )
                queries.append(latest_query)
                continue
            termination = parsed.parser_error_type or "parser_failure"
            break
        return AgentLoopResult(
            conversation=tuple(conversation),
            model_actions=tuple(actions),
            search_queries=tuple(queries),
            termination_reason=termination,
            snapshot_step=int(snapshot_step),
        )
