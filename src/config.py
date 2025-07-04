from pydantic import BaseModel, Field
from typing import List
import yaml,pathlib, os
from logutil import LoggingUtil


class PromptSettings(BaseModel):
    """ Basic class for  prompt location and version"""
    name: str = Field(..., description="Name of the prompt")
    version: str = Field("", description="Version of the prompt")
    # The langchainhub owner address where to pull prompts from for eg "bagel"
    # see : https://smith.langchain.com/hub/bagel
    hub_base: str = Field('bagel', description="Langchain hub org name")

    @classmethod
    def from_str(cls, prompt_address: str) -> "PromptSettings":
        hub_base, name = prompt_address.split('/')
        name, version = prompt_address.split(':')
        return PromptSettings(name=name, version=version, hub_base=hub_base)

    def __str__(self) -> str:
        name = self.name + ':' + self.version if self.version else self.name
        return self.hub_base + '/' + name


class OpenAIConfig(BaseModel):
    llm_model_name: str = Field(default="google/gemma-3-12b-it", description="Name of the model")
    organization: str = Field(default="", description="OPENAI organization")
    access_key: str = Field(default="", description="OPENAI access key")
    url: str = Field(default="http://vllm-server/v1", description="OpenAI compatible endpoint, leave blank for default")
    llm_model_args: dict = Field(default_factory=lambda: {
        "top_p": 0.1,
        "temperature": 0.1
    }, description="Arguments to pass to the model")


class OLLAMAConfig(BaseModel):
    llm_model_name: str = Field(default="llama3.1:latest", description="Name of the model")
    ollama_base_url: str = Field(default="http://ollama:11434", description="URL of the OLLAMA instance")
    llm_model_args: dict = Field(default_factory=dict, description="Arguments to pass to the model")


class Settings(BaseModel):
    prompts: List[PromptSettings] = Field(default=[], description="Prompts to be used in the applicaton")
    openai_config: OpenAIConfig = Field(default=None, description="")
    ollama_config: OLLAMAConfig = Field(default=None, description="")
    langServe: bool = True
    logging_level: str = "DEBUG"



def load_settings():
    yaml_path = pathlib.Path(os.path.dirname(__file__), '..', 'settings.yaml')
    with open(str(yaml_path), 'r') as stream:
        _settings = Settings(**yaml.load(stream, yaml.FullLoader))
    return _settings


# app settings
settings = load_settings()

logger = LoggingUtil.init_logging('bagel', log_file_level=settings.logging_level)