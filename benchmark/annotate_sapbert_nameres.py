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
import logging


def generate_entities(file_name, abbreviations_dict=None):
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


def extract_entities(file_name, abbreviations_dict):
    """
    Extract all entities from the input file and group them by entity text.
    Returns a dictionary: {entity_text: [entity_info_dict, ...]}
    """
    entity_dict = {}
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            pmid = data.get('pmid')
            abbr_map = abbreviations_dict.get(pmid)
            for entity in data['entities']:
                entity_text = entity['text_segment']
                expanded_entity = entity_text
                # Expand abbreviations
                for abbr_key, abbr_def in abbr_map.items():
                    if abbr_key in expanded_entity:
                        expanded_entity = expanded_entity.replace(abbr_key, abbr_def)
                entity_info = {
                    'original_entity': entity_text,
                    'start_index': entity['start_index'],
                    'end_index': entity['end_index'],
                    'pmid': pmid,
                    'medmentions': {
                        'identifier': entity['identifier'],
                        'label': entity.get('identifier_label',""),
                        'biolink_types': entity['biolink_types'] or [],
                        'description': entity.get('description','')
                    }
                }
                if expanded_entity not in entity_dict:
                    entity_dict[expanded_entity] = []
                entity_dict[expanded_entity].append(entity_info)
    return entity_dict


def load_abbreviations(abbreviations_file):
    """Load abbreviations from a JSONL file into a dict keyed by pmid."""
    abbreviations_dict = {}
    if abbreviations_file:
        with open(abbreviations_file, 'r') as abbr_file:
            for line in abbr_file:
                abbr_entry = json.loads(line)
                pmid = abbr_entry.get('pmid')
                abbr_map = abbr_entry.get('abbreviation_map', [])
                if pmid is not None:
                    abbreviations_dict[pmid] = {a["abbreviation"]: a["definition"] for a in abbr_map}
    return abbreviations_dict


async def annotate_medmentions(input_file, output_file,
                               exclude_types=None,
                               chunk_size=7,
                               name_res_url="",
                               sapbert_url="",
                               node_norm_url="",
                               sapbert_count=None,
                               nameres_count=None,
                               count=20,
                               abbreviations_file=None):
    """
        Annotates entities from an input file and writes them to an output file,
        showing a progress bar.
        """
    if exclude_types is None:
        exclude_types = []

    # Load abbreviations if file is provided
    abbreviations_dict = load_abbreviations(abbreviations_file)
    modified_entities =  extract_entities(input_file, abbreviations_dict)
    dump_entity_map(output_file, modified_entities)

    # First pass: count the total number of entities to be processed.
    print("Counting entities to process...")
    #This approach loses the bl_types to exclude, but it could be added into extract_entities if we wanted
    total_entities = len(modified_entities)
    #total_entities = sum(1 for entity in generate_entities(input_file, abbreviations_dict=abbreviations_dict)
    #                     if not any(bl_type in exclude_types for bl_type in entity['biolink_types']))
    print(f"Found {total_entities} entities to annotate.")

    entities_to_annotate = []
    errors = []
    with open(output_file, 'w') as stream, tqdm(total=total_entities, desc="Annotating entities") as pbar:
        for entity in modified_entities:
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
                # Collect errors
                errors.extend([r for r in results if isinstance(r, dict) and r.get("error")])
                # write results to file
                write_annotations_to_file(results, entities_to_annotate, stream)
                pbar.update(len(entities_to_annotate))
                # reset ...
                entities_to_annotate = []
        if len(entities_to_annotate) > 0:
            results = await annotate_chunk(entities_to_annotate,
                                           name_res_url=name_res_url,
                                           sapbert_url=sapbert_url,
                                           node_norm_url=node_norm_url
                                           )
            errors.extend([r for r in results if isinstance(r, dict) and r.get("error")])
            write_annotations_to_file(results, entities_to_annotate, stream)
            pbar.update(len(entities_to_annotate))

    # Write errors to a file
    import os
    base, ext = os.path.splitext(output_file)
    error_file = f"{base}_errors.json"
    with open(error_file, 'w') as ef:
        json.dump(errors, ef, indent=2)

    # After annotation is done, dump the entity map


def dump_entity_map(output_file, ents):
    """
    Extract entities and write the entity map to a file named like the output file but with _entity_map.json.
    """
    import os
    base, ext = os.path.splitext(output_file)
    entity_map_filename = f"{base}_entity_map.json"
    with open(entity_map_filename, 'w') as f:
        json.dump(ents, f, indent=2)


async def annotate_chunk(list_of_entities , name_res_url, sapbert_url, node_norm_url, count=20, sapbert_count=None, nameres_count= None):
    import logging
    async with httpx.AsyncClient(timeout=600) as client:
        tasks = [
            get_entity_ids(
                entity,
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
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Handle exceptions gracefully
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Annotation failed for entity: {list_of_entities[i]} with error: {result}")
                results[i] = {"error": str(result), "entity": list_of_entities[i]}
        return results


def write_annotations_to_file(annotation_results, entities, stream):
    for annotations, entity in zip(annotation_results, entities):
        annotations["annotated_text"] = entity
        stream.write(json.dumps(annotations) + '\n')

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
    parser.add_argument(
        '--abbreviations-file',
        type=str,
        default=None,
        help='Path to abbreviations JSON file.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name to check in the results file.'
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
            chunk_size=args.chunk_size,
            abbreviations_file=args.abbreviations_file
        )
    )
