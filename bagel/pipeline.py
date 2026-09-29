"""The pipeline linking strategy: prepare -> retrieve -> adjudicate -> select, for one mention."""
import httpx
from pydantic import BaseModel, Field

from bagel.abbreviations import expand
from bagel.adjudicate import TaskSpec, adjudicate
from bagel.backends import Backend
from bagel.cache import Cache
from bagel.retrieval import RetrievalConfig, retrieve
from bagel.select import SelectionPolicy, select
from bagel.types import Document, Mention, MentionResult, Selection


class PipelineConfig(BaseModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    policy: SelectionPolicy = Field(default_factory=SelectionPolicy)
    use_type_hint: bool = False  # filter retrieval by the mention's type hint


def link(document: Document, mention: Mention, abbreviations: dict[str, str], task: TaskSpec, backend: Backend,
         config: PipelineConfig, client: httpx.Client, cache: Cache | None = None) -> MentionResult:
    """
    Links one mention. Retrieval failures (HTTP errors from NameRes/SapBERT/NodeNorm) are recorded as
    status "retrieval_error" with the message; any other exception is a bug and propagates.
    """
    query = expand(mention.text, abbreviations)
    type_hint = mention.type_hint if config.use_type_hint else None
    try:
        candidates = retrieve(client, config.retrieval, query, type_hint, cache)
    except httpx.HTTPError as e:
        return MentionResult(mention=mention, query=query, status="retrieval_error", error=f"{type(e).__name__}: {e}",
                             selection=Selection(status="failed", reason="retrieval_error"))
    if not candidates:
        return MentionResult(mention=mention, query=query, status="retrieval_empty",
                             selection=Selection(status="abstained", reason="no candidates"))
    decision = adjudicate(task, backend, document, mention, candidates)
    selection = select(decision, candidates, config.policy)
    return MentionResult(mention=mention, query=query, status=decision.status, candidates=candidates,
                         decision=decision, selection=selection, error=decision.error)
