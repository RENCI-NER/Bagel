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
from tqdm import tqdm


def generate_entities(file_name):
    """Generator function to yield entities from the input file."""
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


async def annotate_medmentions(input_file, output_file,
                               exclude_types=None,
                               chunk_size=7,
                               name_res_url="",
                               sapbert_url="",
                               node_norm_url="",
                               sapbert_count=None,
                               nameres_count=None,
                               count=20):
    """
        Annotates entities from an input file and writes them to an output file,
        showing a progress bar.
        """
    if exclude_types is None:
        exclude_types = []

    # First pass: count the total number of entities to be processed.
    print("Counting entities to process...")
    total_entities = sum(1 for entity in generate_entities(input_file)
                         if not any(bl_type in exclude_types for bl_type in entity['biolink_types']))
    print(f"Found {total_entities} entities to annotate.")


    entities_to_annotate = []

    with open(output_file, 'w') as stream, tqdm(total=total_entities, desc="Annotating entities") as pbar:
        for entity in generate_entities(input_file):
            if any(bl_type in exclude_types for bl_type in entity['biolink_types']):
                continue
            entities_to_annotate.append(entity)
            if len(entities_to_annotate) == chunk_size:
                # send this off to annotations
                results = await annotate_chunk(entities_to_annotate,
                                               name_res_url=name_res_url,
                                               sapbert_url=sapbert_url,
                                               node_norm_url=node_norm_url,
                                               sapbert_count=sapbert_count,
                                               nameres_count=nameres_count,
                                               count=count)
                # write results to file
                write_annotations_to_file(results, entities_to_annotate, stream)
                pbar.update(len(entities_to_annotate))
                # reset ...
                entities_to_annotate = []
        # and for the remaining ones
        if len(entities_to_annotate) > 0:
            results = await annotate_chunk(entities_to_annotate,
                                           name_res_url=name_res_url,
                                           sapbert_url=sapbert_url,
                                           node_norm_url=node_norm_url
                                           )
            write_annotations_to_file(results, entities_to_annotate, stream)
            pbar.update(len(entities_to_annotate))


async def annotate_chunk(list_of_entities , name_res_url, sapbert_url, node_norm_url, count=20, sapbert_count=None, nameres_count= None):
    async with httpx.AsyncClient(timeout=600) as client:
        tasks = [
            get_entity_ids(
                entity['entity'],
                name_res_url=name_res_url,
                sapbert_url=sapbert_url,
                node_norm_url=node_norm_url,
                session=client,
                entity_type=None,
                count=count,
                sapbert_count=sapbert_count,
                nameres_count=nameres_count
            ) for entity in list_of_entities
        ]
        return await asyncio.gather(*tasks)


def write_annotations_to_file(annotation_results, entities, stream):
    for annotations, entity in zip(annotation_results, entities):
        entity['annotations'] = annotations
        stream.write(json.dumps(entity) + '\n')

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Annotate MedMentions from a given input file.")

    # Add the arguments
    parser.add_argument(
        '--input-file',
        type=str,
        default='../../scratch/corpus_pubtator.jsonl',
        help='The path to the input JSONL file.'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='../../scratch/sap_bert_nameres_annotations.jsonl',
        help='The path to the output JSONL file.'
    )
    parser.add_argument(
        '--exclude-types',
        nargs='*',
        default=['biolink:Protein'],
        help='A list of types to exclude from annotation.'
    )
    parser.add_argument(
        '--name-res-url',
        type=str,
        default="https://name-resolution-sri.renci.org/lookup?autocomplete=false&offset=0&string=",
        help='The URL for the name resolution service.'
    )
    parser.add_argument(
        '--sapbert-url',
        type=str,
        default="https://sap-qdrant.apps.renci.org/annotate/",
        help='The URL for the Sapbert annotation service.'
    )
    parser.add_argument(
        '--node-norm-url',
        type=str,
        default="https://nodenormalization-sri.renci.org/get_normalized_nodes",
        help='The URL for the node normalization service.'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=20,
        help='The number of items to process.'
    )
    parser.add_argument(
        '--sapbert-count',
        type=int,
        default=None,
        help='The count for Sapbert processing.'
    )
    parser.add_argument(
        '--nameres-count',
        type=int,
        default=None,
        help='The count for NameRes processing.'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=7,
        help='How many entites to annotate at a time'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the main async function with the parsed arguments
    asyncio.run(
        annotate_medmentions(
            input_file=args.input_file,
            output_file=args.output_file,
            exclude_types=args.exclude_types,
            name_res_url=args.name_res_url,
            sapbert_url=args.sapbert_url,
            node_norm_url=args.node_norm_url,
            count=args.count,
            sapbert_count=args.sapbert_count,
            nameres_count=args.nameres_count,
            chunk_size=args.chunk_size
        )
    )

