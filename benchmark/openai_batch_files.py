import json
import os.path

import openai
import asyncio

from models import SynonymListContext, Entity
from prompt import load_prompt
from llm_output import parse_llm_json, map_color_codes, LLMOutputError
from glob import glob
from util.ner_util import get_taxa_information

# #############
# # File prep functions
# ###########


def get_abstracts(med_mentions_corpus_file_name):
    """
    Load all abstracts from medmentions input file
    :param med_mentions_corpus_file_name:
    :return: {
    "pmid" : "abstract text"
    }
    """
    with open(med_mentions_corpus_file_name) as file:
        pmid_abstracts = {}
        for line in file:
            line = json.loads(line)
            pmid = line['pmid']
            abstract = line['text']
            pmid_abstracts[pmid] = abstract
    return pmid_abstracts


def create_openai_request(pmid_abstracts, annotations_file_name, prompt, model_name, additional_model_args, nn_url):
    """
    for each entity in annotations file, it yields a json object of an openai request.
    :param pmid_abstracts: used for request identification
    :param annotations_file_name: annotations file name to get entities from
    :param prompt: prompt to use for the request
    :param model_name: model to use
    :param additional_model_args: model args
    :return: openai batch api compatible dict
    """
    with open(annotations_file_name) as stream:
        for line in stream:
            try:
                annotation_obj = json.loads(line)
            except Exception as e:
                print(line)
                raise e
            context = SynonymListContext(
                text=pmid_abstracts[annotation_obj['pmid']],
                entity=annotation_obj['entity'],
                synonyms=[
                    Entity(**{
                        "label": value["name"],
                        "identifier": identifier,
                        "description": value.get("description", ""),
                        "entity_type": value.get("category", ""),
                        "taxa": ", ".join(
                            [value for key, value in asyncio.run(get_taxa_information(value["taxa"], nn_url)).items()]
                        )
                    }) for identifier, value in annotation_obj['annotations'].items()
                ]
            )
            prompt_message = prompt.format_prompt(**{
                'text': context.text,
                'term': context.entity,
                'synonym_list': context.pretty_print_synonyms()
            })
            request_id = f"{annotation_obj['pmid']}-{annotation_obj['start_index']}-{annotation_obj['end_index']}"
            body = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": message.type if message.type != "human" else "user",
                            "content": message.content
                        } for message in prompt_message.messages
                    ],
                # This seems to return a single object instead of a list ...
                    # "response_format": {
                    #     "type": "json_object"
                    # }
                }
            if additional_model_args:
                body.update(additional_model_args)

            yield {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }


def create_openai_batch_files(abstracts_file, annotations_file, output_path, prompt_name, model_name, model_args,
                              chunk_size=50_000, nn_url=""):
    """
    Splits annotation file into entries of chunks of size chunk_size , ready for openai batch api.
    :param abstracts_file: file to get abstracts for context
    :param annotations_file: file containing sapbert and nameres annotaton for entites
    :param output_path: output path of chunked openai batch files
    :param prompt_name: prompt to load from langhub
    :param model_name: model to use for this batch
    :param model_args: args for model
    :param chunk_size: number of records in each batch.
    :return:
    """
    # 50K by default per file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    abstracts_by_pmid = get_abstracts(abstracts_file)
    prompt = load_prompt(prompt_name)
    count = 0
    current_batch = count // chunk_size
    # each file is named outputdir/batch-x.jsonl where x is the index of the batch
    get_file_name = lambda c: os.path.join(output_path, "batch-" + str(c) + ".jsonl")
    req_ids_in_file = []
    file_stream = open(get_file_name(current_batch), 'w')
    for i in create_openai_request(
            abstracts_by_pmid,
            annotations_file_name=annotations_file,
            prompt=prompt,
            model_name=model_name,
            additional_model_args=model_args,
            nn_url=nn_url
        ):
        custom_id = i['custom_id']
        if custom_id not in req_ids_in_file:
            req_ids_in_file.append(custom_id)
            if count // chunk_size != current_batch:
                file_stream.close()
                current_batch = count // chunk_size
                req_ids_in_file = []
                file_stream = open(get_file_name(current_batch), 'w')
            file_stream.write(json.dumps(i) + "\n")
            count += 1
    file_stream.close()

# ##########
# # /END file prep
#
# # Start Batch api code ...
# ##########


def upload_file_and_start_batch(file_name, client):
    """
    Uploads a batch file to the openai client and returns the name of the current batch file
    processed, its id in openai after upload and it's assigned batch id in openai batch api.
    :param file_name: batch api compatible file
    :param client: openai client
    :return: dict
    """
    # from openai import OpenAI
    batch_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )
    file_id = batch_file.id
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    job_id = batch_job.id
    return {
        "file_name": file_name,
        "file_id": file_id,
        "batch_job_id": job_id
    }


def start_batches(directory, client):
    """
    Uploads files and starts jobs on openai batch api
    :param directory: directory to upload batch files from
    :param client: openai client
    :return: a file `batch_status.json` is created logging batch id to openai file id to batch file id mapped.
    """
    files = glob("{}/batch-*.jsonl".format(directory))
    batch_data = []
    for file in files:
        print("uploading batch {}".format(file))
        batch_data.append(
            upload_file_and_start_batch(file, client)
        )
        print(f"response: {batch_data}")
    with open(os.path.join(directory, 'batch_status.json'), 'w') as stream:
        json.dump(batch_data, stream, indent=2)


def print_batch_status(directory, client: openai.Client):
    """
    Reads in batch_status.json and makes call to openai api to get status of jobs.
    :param directory: working dir
    :param client: openai dir
    :return:
    """
    with open(os.path.join(directory, 'batch_status.json')) as stream:
        batch_data = json.load(stream)
    for batch_row in batch_data:
        batch = client.batches.retrieve(batch_row["batch_job_id"])
        print(f""
              f"{batch_row['file_name'].split(os.path.sep)[-1]} "
              f"status: {batch.status} "
              f"( complete: {batch.request_counts.completed}) "
              f"(failed: {batch.request_counts.failed}) / total:  {batch.request_counts.total}")


def get_batch_results(directory, client: openai.Client):
    """
    Downloads result file from openai and stores them in directory/error/. or directory/complete/.
    subdirs based on the status from openai
    :param directory: working dir
    :param client: openai client
    :return:
    """
    errors_dir = os.path.join(directory, 'error')
    outputs_dir = os.path.join(directory, 'complete')
    if not os.path.exists(errors_dir):
        os.makedirs(errors_dir)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    with open(os.path.join(directory, 'batch_status.json')) as stream:
        batch_data = json.load(stream)
    for batch_row in batch_data:
        batch = client.batches.retrieve(batch_row["batch_job_id"])
        if batch.error_file_id:
            error_file = os.path.join(errors_dir, batch_row['file_name'].split(os.path.sep)[-1])
            client.files.content(batch.error_file_id).write_to_file(error_file)
            print(f"error file wrote to {error_file}")
        if batch.output_file_id:
            response_file_name = os.path.join(outputs_dir, batch_row['file_name'].split(os.path.sep)[-1])
            client.files.content(batch.output_file_id).write_to_file(response_file_name)
            print(f"batch response file wrote to {response_file_name}")
        if batch.errors:
            errors = batch.errors.data
            with open(os.path.join(errors_dir, batch_row['file_name'].split(os.path.sep)[-1].replace('.jsonl', '.log'))
                    , mode='w') as log_stream:
                log_stream.write('\n'.join([f'{e.code} - {e.message}' for e in errors]))

# #########
# # /End batch call
#
# # results processing
# #########

def remap_all(annotation_file, output_dir):
    files = glob(os.path.join(output_dir, 'complete', 'batch-*.jsonl'))
    remap_dir = os.path.join(output_dir,'remapped')
    os.makedirs(os.path.join(output_dir,'remapped'), exist_ok=True)
    final_stats = {}
    for file in files:
        file_name = file.split(os.path.sep)[-1]
        output_file_name = os.path.join(remap_dir, file_name)
        stats = remap_results(annotation_file, file, output_file_name)
        for k in stats:
            final_stats[k] = final_stats.get(k, 0) + stats[k]
    with open(os.path.join(output_dir, 'remap_stats.json') , 'w') as stream:
        json.dump(final_stats, stream)




def remap_results(annotation_file, batch_result_file, output_file_name):
    """
    Remaps openai results to original annotations.
    :param annotation_file: original annotation file
    :param batch_result_file: openai batch file sent
    :param output_file_name:  output of remapping.
    :return:
    """
    annotations_by_req_id = {}
    stats = {
        "hallucinations": 0,        # records with at least one color code not in the candidate list
        "hallucinated_items": 0,    # individual response items with such a color code
        "unmapped_items": 0,        # other unusable items (missing color code / type, not an object)
        "json_errors": 0,           # records whose response contained no parsable JSON list
        "api_errors": 0,            # records whose batch response was not a successful completion
        "total_lines": 0
    }
    print(f"Remapping : {batch_result_file} to {output_file_name}")
    processed_req_ids = []
    with open(annotation_file) as stream:
        for line in stream:
            annotation_obj = json.loads(line)
            req_id = f"{annotation_obj['pmid']}-{annotation_obj['start_index']}-{annotation_obj['end_index']}"
            annotations_by_req_id[req_id] = annotation_obj
    with open(batch_result_file) as stream:
        for line in stream:
            stats["total_lines"] += 1
            batch_result = json.loads(line)
            req_id = batch_result["custom_id"]
            processed_req_ids.append(req_id)
            annotation = annotations_by_req_id[req_id]
            # Every record gets an explicit status so failures can't be confused with "no match" (empty remap).
            annotation["llm_status"] = "ok"
            annotation["llm_response_remapped"] = {}
            try:
                content = batch_result["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                stats["api_errors"] += 1
                annotation["llm_status"] = "api_error"
                annotation["llm_raw_response"] = batch_result.get("response") or batch_result.get("error")
                continue
            try:
                llm_response = parse_llm_json(content)
            except LLMOutputError:
                stats["json_errors"] += 1
                annotation["llm_status"] = "json_error"
                annotation["llm_raw_response"] = content
                continue
            # map color code to synonym ids; must build the list in the same order as create_openai_request
            entity_list = SynonymListContext(
                text="",
                entity="",
                synonyms=[
                    Entity(**{
                        "label": value["name"],
                        "identifier": identifier,
                        "description": value.get("description", ""),
                        "entity_type": value.get("category", ""),
                    }) for identifier, value in annotation["annotations"].items()
                ]
            )
            color_code_remap, problems = map_color_codes(entity_list.synonyms, llm_response)
            if problems:
                hallucinated = [p for p in problems if p["reason"] == "unknown_color"]
                stats["hallucinated_items"] += len(hallucinated)
                stats["unmapped_items"] += len(problems) - len(hallucinated)
                if hallucinated:
                    stats["hallucinations"] += 1
                annotation["llm_status"] = "partial"
                annotation["llm_unmapped_items"] = problems
                annotation["llm_raw_response"] = content
                print(f"Could not map {len(problems)} item(s) for {req_id}: {problems}")
            remapped_response = {}
            for r in color_code_remap:
                cp = dict(annotation["annotations"][r["identifier"]])
                cp.update({"synonym_type": r["synonym_type"]})
                remapped_response[r["identifier"]] = cp
            annotation["llm_response_remapped"] = remapped_response

    with open(output_file_name, 'w') as stream:
        for req_id in processed_req_ids:
            annotation = annotations_by_req_id[req_id]
            stream.write(json.dumps(annotation) + '\n')
        print(f"done processing {output_file_name}")

    return stats

# ###
# /End result processing
# ######


def main(config, subcommand, oai_client):
    med_mentions_file = config['medmentions_file']
    annotations = config['annotation_file_path']
    output_dir = config['output_dir']
    prompt = config['prompt_name']
    m_args = config['model_args']
    m_name = config['model_name']
    chunk_size = config['chunk_size']
    nn_url = config['node_norm_url']
    if subcommand == 'start':

        create_openai_batch_files(
            abstracts_file=med_mentions_file,
            annotations_file=annotations,
            output_path=output_dir,
            prompt_name=prompt,
            model_name=m_name,
            model_args=m_args,
            chunk_size=chunk_size,
            nn_url=nn_url
        )
        start_batches(
            directory=output_dir,
            client=oai_client
        )

    elif subcommand == 'process':
        try:
            get_batch_results(directory=output_dir, client=oai_client)
            remap_all(annotation_file=annotations, output_dir=output_dir)
        except FileNotFoundError as ex:
            print("Error, could not find batch status file, probably because this batch never run."
                  "Try `start` subcommand to fix.")

    elif subcommand == 'status':
        try:
            print_batch_status(directory=output_dir, client=oai_client)
        except FileNotFoundError as ex:
            print("Error, could not find batch status file, probably because this batch never run."
                  "Try `start` subcommand to fix.")




if __name__ == '__main__':
    import argparse
    from openai import OpenAI

    parser = argparse.ArgumentParser(
        description='Batch Annotation tool for Bagel'
    )
    parser.add_argument('-c', '--config', required=True, help='Config file for annotaton task')

    subparsers = parser.add_subparsers(dest="subparser_name")
    sub_commands = {
        'start': 'create batch files and start process',
        'status': 'Print status of openai batch processing',
        'process': 'Process openai response once complete'
    }
    sub_parsers = {
        x: subparsers.add_parser(x, help=sub_commands[x]) for x in sub_commands
    }

    args = parser.parse_args()

    config_file_path = args.config
    _subcommand = args.subparser_name
    if _subcommand not in sub_commands:
        raise ValueError(f"Unknown subcommand `{_subcommand}` options are one of {list(sub_commands.keys())}")
    with open(config_file_path, 'r') as stream:
        _config = json.load(stream)
    _oai_client = OpenAI()
    main(
        config=_config,
        subcommand=_subcommand,
        oai_client=_oai_client
    )