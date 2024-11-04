###
## This file takes in medmentions file as an input. for every entity in med mentions file it
## will call bagel's entity resolution code defined in src/util/ner_util.py(get_entities) function and stream the output
## into a json file.
##
import argparse
import json
from util.ner_util import get_entity_ids
import httpx
import asyncio


def generate_entities(file_name):
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            for entity in data['entities']:
                yield {
                    'entity': entity['text_segment'],
                    'start_index': entity['start_index'],
                    'end_index': entity['end_index'],
                    'biolink_types': entity['biolink_types'] or [],
                    'pmid': data['pmid'],
                    'identifier': entity['identifier']
                }


async def annotate_medmentions(input_file, output_file, exclude_types=[]):
    entities_to_annotate = []
    chunk_size = 7
    with open(output_file, 'w') as stream:
        for entity in generate_entities(input_file):
            if any(bl_type in exclude_types for bl_type in entity['biolink_types']):
                continue
            entities_to_annotate.append(entity)
            if len(entities_to_annotate) == chunk_size:
                # send this off to annotations
                results = await annotate_chunk(entities_to_annotate)
                # write results to file
                write_annotations_to_file(results, entities_to_annotate, stream)
                # reset ...
                entities_to_annotate = []
        # and for the remaining ones
        if len(entities_to_annotate) > 0:
            results = await annotate_chunk(entities_to_annotate)
            write_annotations_to_file(results, entities_to_annotate, stream)


async def annotate_chunk(list_of_entities):
    async with httpx.AsyncClient(timeout=600) as client:
        tasks = [
            get_entity_ids(
                entity['entity'],
                name_res_url="https://name-resolution-sri.renci.org/lookup?autocomplete=false&offset=0&string=",
                sapbert_url="https://sap-qdrant.apps.renci.org/annotate/",
                node_norm_url="https://nodenormalization-sri.renci.org/get_normalized_nodes",
                session=client,
                entity_type=None,
                count=10
            ) for entity in list_of_entities
        ]
        return await asyncio.gather(*tasks)


def write_annotations_to_file(annotation_results, entities, stream):
    for annotations, entity in zip(annotation_results, entities):
        entity['annotations'] = annotations
        stream.write(json.dumps(entity) + '\n')


if __name__ == "__main__":
    asyncio.run(
        annotate_medmentions(
            input_file='/home/kebedey/projects/ner/scratch/corpus_pubtator.jsonl',
            output_file='/home/kebedey/projects/ner/scratch/sap_bert_nameres_annotations.jsonl',
            exclude_types=['biolink:Protein']
        )
    )



