from pydantic import BaseModel, Field
from typing import List


class Entity(BaseModel):
    """ Entity class: Represents a normalized entity (Synonym)."""
    label: str = Field(..., description="Label of the Entity.")
    identifier: str = Field("", description="Curie identifier of the Entity.")
    description: str = Field("", description="Formal description(definition) of the Entity.")
    entity_type: str = Field("", description="Type of the Entity.")

class SynonymListContext(BaseModel):
    text: str = Field(..., description="Body of text containing entity.")
    entity: str = Field(..., description="Entity identified in text.")
    synonyms: List[Entity] = Field(..., description="Entities linked to the target entity, to be re-ranked.")

    def pretty_print_synonyms(self):
        string = "\n"
        for synonym in self.synonyms:
            string += f"\t- {synonym.label}"
            string += f" ({synonym.entity_type})" if synonym.entity_type else ""
            string += f" ({synonym.identifier})" if synonym.identifier else ""
            string += f" : {synonym.description}" if synonym.description else ""
            string += "\n"
        return string



