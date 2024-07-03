from pydantic import BaseModel, Field
from typing import List


colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "cyan",
    "magenta",
    "lime",
    "maroon",
    "navy",
    "olive",
    "teal",
    "aqua",
    "gold"
]







class Entity(BaseModel):
    """ Entity class: Represents a normalized entity (Synonym)."""
    label: str = Field(..., description="Label of the Entity.")
    identifier: str = Field("", description="Curie identifier of the Entity.")
    description: str = Field("", description="Formal description(definition) of the Entity.")
    entity_type: str = Field("", description="Type of the Entity.")
    color_code: str = Field("", description="Color coding for mapping back items.")

class SynonymListContext(BaseModel):
    text: str = Field(..., description="Body of text containing entity.")
    entity: str = Field(..., description="Entity identified in text.")
    synonyms: List[Entity] = Field(..., description="Entities linked to the target entity, to be re-ranked.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for index, synonym in enumerate(self.synonyms):
            synonym.color_code = colors[index]

    def pretty_print_synonyms(self):
        string = "\n"
        for synonym in self.synonyms:
            string += f"\t- {synonym.label}"
            string += f" ({synonym.entity_type}) ({synonym.color_code})" if synonym.entity_type else ""
            string += f" : {synonym.description}" if synonym.description else ""
            string += "\n"
        return string


class SynonymClassResponse(BaseModel):
    synonym: str = Field(..., description="Synonym")
    vocabulary_class: str = Field(..., description="Vocabulary class")
    synonym_type: str = Field(..., description="Synonym type")


class SynonymClassesResponse(BaseModel):
    synonyms: List[SynonymClassResponse] = Field(..., description="Synonyms")