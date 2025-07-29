from pydantic import BaseModel, Field
from typing import List


colors = ['alizarin', 'amaranth', 'amber', 'amethyst', 'apricot', 'aqua', 'aquamarine', 'asparagus', 'auburn', 'azure', 'beige', 'bistre', 'black', 'blue', 'blue-green', 'blue-violet', 'bondi-blue', 'brass', 'bronze', 'brown', 'buff', 'burgundy', 'camouflage-green', 'caput-mortuum', 'cardinal', 'carmine', 'carrot-orange', 'celadon', 'cerise', 'cerulean', 'champagne', 'charcoal', 'chartreuse', 'cherry-blossom-pink', 'chestnut', 'chocolate', 'cinnabar', 'cinnamon', 'cobalt', 'copper', 'coral', 'corn', 'cornflower', 'cream', 'crimson', 'cyan', 'dandelion', 'denim', 'ecru', 'emerald', 'eggplant', 'falu-red', 'fern-green', 'firebrick', 'flax', 'forest-green', 'french-rose', 'fuchsia', 'gamboge', 'gold', 'goldenrod', 'green', 'grey', 'han-purple', 'harlequin', 'heliotrope', 'hollywood-cerise', 'indigo', 'ivory', 'jade', 'kelly-green', 'khaki', 'lavender', 'lawn-green', 'lemon', 'lemon-chiffon', 'lilac', 'lime', 'lime-green', 'linen', 'magenta', 'magnolia', 'malachite', 'maroon', 'mauve', 'midnight-blue', 'mint-green', 'misty-rose', 'moss-green', 'mustard', 'myrtle', 'navajo-white', 'navy-blue', 'ochre', 'office-green', 'olive', 'olivine', 'orange', 'orchid', 'papaya-whip', 'peach', 'pear', 'periwinkle', 'persimmon', 'pine-green', 'pink', 'platinum', 'plum', 'powder-blue', 'puce', 'prussian-blue', 'psychedelic-purple', 'pumpkin', 'purple', 'quartz-grey', 'raw-umber', 'razzmatazz', 'red', 'robin-egg-blue', 'rose', 'royal-blue', 'royal-purple', 'ruby', 'russet', 'rust', 'safety-orange', 'saffron', 'salmon', 'sandy-brown', 'sangria', 'sapphire', 'scarlet', 'school-bus-yellow', 'sea-green', 'seashell', 'sepia', 'shamrock-green', 'shocking-pink', 'silver', 'sky-blue', 'slate-grey', 'smalt', 'spring-bud', 'spring-green', 'steel-blue', 'tan', 'tangerine', 'taupe', 'teal', 'tenné-(tawny)', 'terra-cotta', 'thistle', 'titanium-white', 'tomato', 'turquoise', 'tyrian-purple', 'ultramarine', 'van-dyke-brown', 'vermilion', 'violet', 'viridian', 'wheat', 'white', 'wisteria', 'yellow', 'zucchini']







class Entity(BaseModel):
    """ Entity class: Represents a normalized entity (Synonym)."""
    label: str = Field(..., description="Label of the Entity.")
    identifier: str = Field("", description="Curie identifier of the Entity.")
    description: str = Field("", description="Formal description(definition) of the Entity.")
    entity_type: str = Field("", description="Type of the Entity.")
    color_code: str = Field("", description="Color coding for mapping back items.")
    taxa: str = Field("", description="Taxonomic label of the Entity.")
    taxa_ids: list = Field([], description="Taxonomic identifiers of the Entity.")

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
            string += f" [{synonym.taxa}] " if synonym.taxa else ""
            string += f"({synonym.entity_type}) ({synonym.color_code})" if synonym.entity_type else ""
            string += f" : {synonym.description}" if synonym.description else ""
            string += "\n"
        return string


class SynonymClassResponse(BaseModel):
    synonym: str = Field(..., description="Synonym")
    vocabulary_class: str = Field(..., description="Vocabulary class")
    synonym_type: str = Field(..., description="Synonym type")


class SynonymClassesResponse(BaseModel):
    synonyms: List[SynonymClassResponse] = Field(..., description="Synonyms")