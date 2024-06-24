# set the LANGCHAIN_API_KEY environment variable (create key in settings)
from langchain import hub
from config import PromptSettings, List


def load_prompt_from_hub(prompt_name):
    """ Loads prompts from langchain hub and returns a prompt object."""
    return hub.pull(str(prompt_name))


def load_prompts(prompts: List[PromptSettings]):
    # loads all prompts
    prompt_mapping = {}
    for prompt in prompts:
        prompt_mapping[prompt.name] = load_prompt_from_hub(str(prompt))
    return prompt_mapping

