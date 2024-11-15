## Configuration
The tool relies on a configuration file to define the workflow parameters. Below is an example configuration file:

```json
{
  "model_name": "gpt-4o",
  "model_args": {
    "temperature": 0
  },
  "prompt_name": "bagel/ask_classes:4da4ad77",
  "output_dir": "/path/to/output/directory/",
  "annotation_file_path": "/path/to/annotations.jsonl",
  "medmentions_file": "/path/to/med_mentions.jsonl"
}
```
#### Key Configuration Fields
- model_name: The OpenAI model to use (e.g., gpt-4o).
- model_args: Additional parameters for the model (e.g., temperature).
- prompt_name: The prompt identifier to use for OpenAI completions.
- output_dir: Path to store batch files and outputs.
- annotation_file_path: Path to the annotation file containing entities and identifiers.
- medmentions_file: Path to the file with abstracts or textual data.
#### Usage
Run the tool using the following commands:

Start a New Batch Process:
```bash

python openai_batch_files.py -c config.json start
```
##### Check Batch Processing Status:
```bash

python openai_batch_files.py -c config.json status
```
##### Process Completed Batches:
```bash

python openai_batch_files.py -c config.json process
```

##### File Structure
###### Input Files

* Abstract File: 
Contains abstracts for context during annotation.

* Annotation File:
Contains entities and their annotations.

###### Output Files
* Batch Files: 
Split JSONL files for OpenAI batch API.

* Remapped Files: 
Processed outputs remapped to the original annotations.