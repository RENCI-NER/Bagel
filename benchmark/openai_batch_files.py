import json
import os.path

import openai

from models import SynonymListContext, Entity
from prompt import load_prompt_from_hub
from chain import LLMHelper
from glob import glob

##############
## File prep functions
############


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


def create_openai_request(pmid_abstracts, annotations_file_name, prompt, model_name, additional_model_args):
    """
    for each entity in annotations file, it yields a json object of an openai request.
    :param pmid_abstracts: used for request identification
    :param annotations_file_name: annotations file name to get entities from
    :param prompt: prompt to use for the request
    :param model_name: model to use
    :param additional_model_args: model args
    :return: openai batch api compatible dict
    """
    with (open(annotations_file_name) as stream):
        for line in stream:
            annotation_obj = json.loads(line)
            context = SynonymListContext(
                text=pmid_abstracts[annotation_obj['pmid']],
                entity=annotation_obj['entity'],
                synonyms=[
                    Entity(**{
                        "label": value["name"],
                        "identifier": identifier,
                        "description": value.get("description", ""),
                        "entity_type": value.get("category", ""),
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


def create_openai_batch_files(abstracts_file, annotations_file, output_path, prompt_name, model_name, model_args, chunk_size=50_000):
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
    prompt = load_prompt_from_hub(prompt_name)
    count = 0
    current_batch = count // chunk_size
    # each file is named outputdir/batch-x.jsonl where x is the index of the batch
    get_file_name = lambda c: os.path.join(output_path, "batch-" + str(c) + ".jsonl")
    file_stream = open(get_file_name(current_batch), 'w')
    for i in create_openai_request(
            abstracts_by_pmid,
            annotations_file_name=annotations_file,
            prompt=prompt,
            model_name=model_name,
            additional_model_args=model_args
        ):
        if count // chunk_size != current_batch:
            file_stream.close()
            current_batch = count // chunk_size
            file_stream = open(get_file_name(current_batch), 'w')
        file_stream.write(json.dumps(i) + "\n")
        count += 1
    file_stream.close()

###########
## /END file prep
#
## Start Batch api code ...
###########


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
        if batch.status == "failed":
            error_file = os.path.join(errors_dir, batch_row['file_name'].split(os.path.sep)[-1])

            client.files.content(batch.error_file_id).write_to_file(error_file)
            print(f"error file wrote to {error_file}")
        elif batch.status == "completed":
            response_file_name = os.path.join(outputs_dir, batch_row['file_name'].split(os.path.sep)[-1])
            client.files.content(batch.output_file_id).write_to_file(response_file_name)
            print(f"batch response file wrote to {response_file_name}")
        else:
            print(f"batch status {batch.status} , for file {batch_row['file_name']}")

##########
## /End batch call
#
## results processing
##########

def remap_all(annotation_file, output_dir):
    files = glob(os.path.join(output_dir, 'complete', 'batch-*.jsonl'))
    remap_dir = os.path.join(output_dir,'remapped')
    os.makedirs(os.path.join(output_dir,'remapped'), exist_ok=True)
    for file in files:
        file_name = file.split(os.path.sep)[-1]
        output_file_name = os.path.join(remap_dir, file_name)
        remap_results(annotation_file, file, output_file_name)




def remap_results(annotation_file, batch_result_file, output_file_name):
    """
    Remaps openai results to original annotations.
    :param annotation_file: original annotation file
    :param batch_result_file: openai batch file sent
    :param output_file_name:  output of remapping.
    :return:
    """
    annotations_by_req_id = {}
    with open(annotation_file) as stream:
        for line in stream:
            annotation_obj = json.loads(line)
            req_id = f"{annotation_obj['pmid']}-{annotation_obj['start_index']}-{annotation_obj['end_index']}"
            annotations_by_req_id[req_id] = annotation_obj
    with (open(batch_result_file) as stream):
        for line in stream:
            batch_result = json.loads(line)
            req_id = batch_result["custom_id"]
            llm_response = json.loads(batch_result["response"]["body"]["choices"][0]["message"]["content"]
                                      .replace('```json', '')
                                      .replace('```', '')) #if isinstance(batch_result["response"]["body"]["choices"][0]["message"], str) else batch_result["response"]["body"]["choices"][0]["message"]
            annotations_by_req_id[req_id]["llm_repsonse"] = llm_response
            # map color code to synonym ids
            entity_list = SynonymListContext(
                text="",
                entity="",
                synonyms=[
                    Entity(**{
                        "label": value["name"],
                        "identifier": identifier,
                        "description": value.get("description", ""),
                        "entity_type": value.get("category", ""),
                    }) for identifier, value in annotations_by_req_id[req_id]["annotations"].items()
                ]
            )
            color_code_remap = LLMHelper.re_map_responses(entity_list.synonyms, llm_response)
            remapped_response = {}
            for r in color_code_remap:
                cp = {key: val for key, val in annotations_by_req_id[req_id]["annotations"][r["identifier"]].items()}
                cp.update({"synonym_type": r["synonym_type"]})
                remapped_response[r["identifier"]] = cp
            annotations_by_req_id[req_id]["llm_response_remapped"] = remapped_response

    with open(output_file_name, 'w') as stream:
        for req_id, annotation in annotations_by_req_id.items():
            stream.write(json.dumps(annotation) + '\n')

####
# /End result processing
#######


def main(config, subcommand, oai_client):
    med_mentions_file = config['medmentions_file']
    annotations = config['annotation_file_path']
    output_dir = config['output_dir']
    prompt = config['prompt_name']
    m_args = config['model_args']
    m_name = config['model_name']
    if subcommand == 'start':
        create_openai_batch_files(
            abstracts_file=med_mentions_file,
            annotations_file=annotations,
            output_path=output_dir,
            prompt_name=prompt,
            model_name=m_name,
            model_args=m_args,
            chunk_size=50_000
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
        'start':'create batch files and start process',
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
    _oai_client = OpenAI(
    )
    main(
        config=_config,
        subcommand=_subcommand,
        oai_client=_oai_client
    )


    # ann_file =
    # abstract_file = '../../scratch/corpus_pubtator.jsonl'
    # output_path = '../../scratch/batch-1/'
    # model_args = {
    #     "temperature": 0
    # }
    # model_name = "gpt-4o"
    # prompt_name = ''
    # create_openai_batch_files(
    #     abstracts_file=abstract_file,
    #     annotations_file=ann_file,
    #     output_path=output_path,
    #     prompt_name=prompt_name,
    #     model_name=model_name,
    #     model_args=model_args,
    #     chunk_size=3
    # )





    # start_batches(output_path, client=client)
    # print_batch_status(output_path, client)
    # get_batch_results(output_path, client)
    # remap_all(ann_file, output_path)
    # abstracts = get_abstracts(abstracts_file)
    # prompt = load_prompt_from_hub()
    # openai_batch_file_name = '../../scratch/batch_test_file.json'
    # with open(openai_batch_file_name, 'w') as stream:
    #     for x in create_openai_request(abstracts, ann_file, prompt,
    #                                    model_name='gpt-4o',
    #                                    additional_model_args={'temperature': 0}
    #                                    ):
    #         stream.write(json.dumps(x) + '\n')
    # batch_job = upload_file_and_start_batch(file_name=openai_batch_file_name,
    #                                         client=client)
    # with open("../../scratch/batch_log_test.json", 'w') as stream:
    #     json.dump(batch_job.to_json(), stream, indent=2)

    # client.files.content('file-tbxvGcyw2UoWAUtUgdkq9OLq').stream_to_file("../../scratch/openai-response-2.jsonl")
    #
    # client.files.content("file-WbxyxVmqMTBt2mF96UHVmt58")
    # remap_results(ann_file, "../../scratch/openai-response.jsonl", "../../scratch/openai-remapped.jsonl" )


