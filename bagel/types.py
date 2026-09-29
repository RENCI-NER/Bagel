"""Data passed between the linking stages."""
from enum import StrEnum

from pydantic import BaseModel, Field


class Relation(StrEnum):
    """SKOS mapping relation, with the mention as subject and the candidate as object."""
    EXACT = "exactMatch"
    BROAD = "broadMatch"  # the candidate is broader than the mention
    NARROW = "narrowMatch"  # the candidate is narrower than the mention
    RELATED = "relatedMatch"
    NONE = "noMatch"


class Document(BaseModel):
    id: str
    text: str


class Mention(BaseModel):
    text: str
    start: int | None = None
    end: int | None = None
    type_hint: str | None = None  # Biolink category, e.g. "biolink:Disease"


class Candidate(BaseModel):
    curie: str
    label: str
    category: str = ""
    description: str = ""
    taxa: list[str] = Field(default_factory=list)  # taxon labels
    taxa_ids: list[str] = Field(default_factory=list)
    source_curies: list[str] = Field(default_factory=list)  # retrieved CURIEs that normalized to this one
    name_res_rank: int | None = None
    name_res_score: float | None = None
    sapbert_rank: int | None = None
    sapbert_score: float | None = None

    @property
    def best_rank(self) -> int | None:
        ranks = [r for r in (self.name_res_rank, self.sapbert_rank) if r is not None]
        return min(ranks) if ranks else None


class Judgment(BaseModel):
    curie: str
    relation: Relation
    confidence: float | None = None


class Decision(BaseModel):
    """Result of the adjudication stage for one mention."""
    status: str  # ok | api_error | parse_error | invalid_candidate | incomplete
    judgments: list[Judgment] = Field(default_factory=list)
    rationale: str | None = None
    error: str | None = None
    raw: str | None = None
    usage: dict = Field(default_factory=dict)
    seconds: float | None = None

    def with_relation(self, relation: Relation) -> list[Judgment]:
        return [j for j in self.judgments if j.relation == relation]


class Selection(BaseModel):
    status: str  # linked | abstained | failed
    curie: str | None = None
    relation: Relation | None = None
    confidence: float | None = None
    reason: str | None = None


class MentionResult(BaseModel):
    """Everything BAGEL produced for one mention; what a run stores per mention."""
    mention: Mention
    query: str
    status: str  # ok, or the status of the stage that failed
    candidates: list[Candidate] = Field(default_factory=list)
    decision: Decision | None = None
    selection: Selection
    error: str | None = None
