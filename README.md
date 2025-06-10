# Bagel API Server
The Bagel API Server is a FastAPI application designed to run re-ranking prompts against pre-configured Large Language Models (LLMs). Its primary function is to enhance entity normalization by using LLMs to classify and re-rank a list of potential candidate identifiers (CURIEs) for a given entity within a text.

The server supports both OpenAI and Ollama models and features two main workflows:

1. `find_curies`: Automatically discovers candidate identifiers from external services and then uses an LLM to re-rank them.

2. `group_synonyms`: Re-ranks a user-provided list of candidate synonyms using an LLM.

Prompts are dynamically loaded from the LangChain Community Hub, allowing for flexible and updatable prompt engineering without changing the server code.


## Setup and Installation

### Prerequisites
* Python 3.8+
* An active OpenAI API key (for OpenAI endpoints).
* Ollama installed and running with a desired model (e.g., ollama run llama3) for Ollama endpoints.

### Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Running the Server
Start the server using `uvicorn`:

```bash
uvicorn src.server:app --host 0.0.0.0 --port 9001
```

The API documentation will be available at `http://localhost:9001/docs`.

## API Endpoints
The server exposes four main endpoints for LLM-augmented re-ranking.

Note: Prompts are specified by prompt_name and are pulled from https://smith.langchain.com/hub. There are prompts that we have built that could readily be used [here](https://smith.langchain.com/hub/bagel/?organizationId=c9d09f57-ad7a-5cce-8c42-eaf249162e49)

### 1. Find & Rank CURIEs
These endpoints first query external services (Name Resolver, Sapbert) to generate a list of candidate identifiers for an entity and then use an LLM to re-rank them based on the provided text.

`POST /find_curies_openai`
* **Description:** Uses an OpenAI model to find and re-rank candidate identifiers.

* **Request Body OpenAI:**
```json
{
  "prompt_name": "bagel/ask_classes_no_system",
  "text": "The study focused on the effects of Aspirin on heart disease.",
  "entity": "Aspirin",
  "entity_type": "biolink:ChemicalEntity",
  "config": {
    "model": "gpt-4o", 
    "organization": "xxx" , 
    "access_key" : "xxx", 
    "llm_model_args": {}
  }
}
```

`POST /find_curies_ollama`
* **Description:** Uses an Ollama model to find and re-rank candidate identifiers.
* **Request Body:**
```json
{
  "prompt_name": "bagel/ask_classes_no_system",
  "text": "The study focused on the effects of Aspirin on heart disease.",
  "entity": "Aspirin",
  "entity_type": "biolink:ChemicalEntity",
  "config": {
    "model": "llama3.1",
    "ollama_base_url": "https://ollama.apps.renci.org",
    "llm_model_args": {}
  }
}

```


### 2. Group & Rank Synonyms
These endpoints re-rank a user-provided list of synonyms/identifiers for an entity. They do not perform the initial search for candidates.


`POST /group_synonyms_openai`

* **Description:** Uses an OpenAI model to re-rank a provided list of synonyms.
* **Request Body:**
```json
{
  "prompt_name": "bagel/ask_classes_no_system",
  "context": {
    "text": "The study focused on the effects of Aspirin on heart disease.",
    "entity": "Aspirin",
    "synonyms": [
      {
        "label": "Aspirin",
        "identifier": "CHEBI:15365",
        "description": "A small molecule drug."
      },
      {
        "label": "Acetylsalicylic acid",
        "identifier": "DRUGBANK:DB00945",
        "description": "The active ingredient in Aspirin."
      }
    ]
  },
  "config": {
    "model": "gpt-4o", 
    "organization": "xxx" , 
    "access_key" : "xxx", 
    "llm_model_args": {}
  }
}
```

`POST /group_synonyms_ollama`

* **Description:** Uses an Ollama model to re-rank a provided list of synonyms.
* **Request Body:**

```json
{
  "prompt_name": "bagel/ask_classes_no_system",
  "context": {
    "text": "The study focused on the effects of Aspirin on heart disease.",
    "entity": "Aspirin",
    "synonyms": [
      {
        "label": "Aspirin",
        "identifier": "CHEBI:15365",
        "description": "A small molecule drug."
      },
      {
        "label": "Acetylsalicylic acid",
        "identifier": "DRUGBANK:DB00945",
        "description": "The active ingredient in Aspirin."
      }
    ]
  },
  "config": {
    "model": "llama3.1",
    "ollama_base_url": "https://ollama.apps.renci.org",
    "llm_model_args": {}
  }
}
```


#### Example Response Format
All endpoints return a JSON object where keys are the identifiers and values are the re-ranked entity details, including the classification from the LLM (e.g., synonym_type).
```json

{
  "CHEBI:15365": {
    "name": "Aspirin",
    "category": "biolink:ChemicalEntity",
    "description": "A small molecule drug.",
    "synonym_type": "EXACT"
  },
  "DRUGBANK:DB00945": {
    "name": "Acetylsalicylic acid",
    "category": "biolink:ChemicalEntity",
    "description": "The active ingredient in Aspirin.",
    "synonym_type": "RELATED"
  }
}

```