from enum import Enum

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoModel

from .fastchat import FastChatController
from .logger import root_logger
from .mock_tokenizer import MockTokenizer
from .wrappers.dev_model import DevModel
from .wrappers.hf_api_model import HFAPIModel
from .wrappers.openai_api_model import OpenAIAPIModel


class ModelSrc(Enum):
    """Valid sources for loading and running models"""

    AUTO = "auto"
    """Placeholder value to automatically decide the model source"""

    HF_LOCAL = "huggingface_local"
    """Models are downloaded locally and run on a GPU"""

    HF_API = "huggingface_hub"
    """Models are run in the cloud through the Huggingface API"""

    OPENAI_API = "openai"
    """Models are run in the cloud through the OpenAI API or run locally using fastchat"""

    DEV = "dev"
    """Models are run locally using manual input or predetermined algorithms.  Used for testing and development purposes"""


class ModelInfo:

    def __init__(self, pretrained_model_name_or_path: str, model_src: ModelSrc, model_class: PreTrainedModel | None = AutoModelForCausalLM, tokenizer_class: PreTrainedTokenizer | None = AutoTokenizer, model_task: str = None):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_src = model_src
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.model_task = model_task

        if self.model_src == ModelSrc.HF_API and self.model_task is None:
            raise ValueError("A model task is required to use HuggingFace models")

    def as_dict(self):
        return vars(self)

    def __eq__(self, other):
        if isinstance(other, ModelInfo):
            return self.as_dict() == other.as_dict()
        else:
            return False


def model_info_from_name(target_model_name: str, model_src: ModelSrc = ModelSrc.AUTO) -> ModelInfo:
    """Gets information for creating a framework model from the name of the underlying model. Agnostic of which framework model this is being used for

    Args:
        target_model_name: The name of the underlying model to use
        model_src: The suggested source of the model to load. Defaults to AUTO
    Returns:
        A ModelInfo instance containing the information necessary to find the given model"""

    if model_src == ModelSrc.AUTO:
        if target_model_name.startswith("dev/"):
            model_src = ModelSrc.DEV
        elif target_model_name.startswith("openai/"):
            model_src = ModelSrc.OPENAI_API
        else:
            model_src = ModelSrc.HF_LOCAL

    match model_src:
        case ModelSrc.DEV:
            return ModelInfo(target_model_name, ModelSrc.DEV, None, None)
        case ModelSrc.HF_API:
            return ModelInfo(target_model_name, ModelSrc.HF_API, None, None)
        case ModelSrc.OPENAI_API:
            model_info = ModelInfo(target_model_name, ModelSrc.OPENAI_API, None, None)
            if target_model_name.startswith("openai/"):
                return model_info

            # Initialize fastchat for inference if it is available and enabled
            if FastChatController.is_available() and FastChatController.is_enabled():
                FastChatController.open(target_model_name)
            else:
                raise RuntimeError(f"FastChatController cannot be opened. Available: {FastChatController.is_available()}, Enabled: {FastChatController.is_enabled()}")
            return model_info
        case ModelSrc.HF_LOCAL:
            tokenizer_class = LlamaTokenizer if target_model_name.startswith("meta-llama/") else AutoTokenizer
            model_class = LlamaForCausalLM if target_model_name.startswith("meta-llama/") else AutoModelForCausalLM
            return ModelInfo(target_model_name, ModelSrc.HF_LOCAL, model_class, tokenizer_class)


def pretrained_from_info(model_info: ModelInfo) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Gets the pretrained model and tokenizer from the given model information

    Args:
        model_info: The model and tokenizer information to get the pretrained model and tokenizer
    Returns:
        A transformers pretrained model and tokenizer for usage within the framework"""

    # Appropriately assign any AUTO model info
    if model_info.model_src == ModelSrc.AUTO:
        model_info = model_info_from_name(model_info.pretrained_model_name_or_path)

    root_logger.debug(f"Loading a pretrained model {model_info.pretrained_model_name_or_path} from {model_info.model_src}")

    match model_info.model_src:
        case ModelSrc.OPENAI_API:
            return OpenAIAPIModel(model_info.pretrained_model_name_or_path), MockTokenizer(model_info.pretrained_model_name_or_path)
        case ModelSrc.HF_API:
            return HFAPIModel(model_info.pretrained_model_name_or_path, model_info.model_task), MockTokenizer(model_info.pretrained_model_name_or_path)
        case ModelSrc.DEV:
            return DevModel(model_info.pretrained_model_name_or_path), MockTokenizer(model_info.pretrained_model_name_or_path)
        case ModelSrc.HF_LOCAL:
            try:
                model = model_info.model_class.from_pretrained(model_info.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
            except ValueError as e:
                root_logger.warning(f"Could not load {model_info.pretrained_model_name_or_path} as a {model_info.model_class} model.  Using AutoModel instead.")
                model = AutoModel.from_pretrained(model_info.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)

            return model, model_info.tokenizer_class.from_pretrained(model_info.pretrained_model_name_or_path)


def pretrained_from_name(model_name: str, model_src: ModelSrc = ModelSrc.AUTO):
    """Gets the pretrained model and tokenizer from the given model name

    Args:
        model_name: The name of the underlying model to use
        model_src: The suggested source of the model to load. Defaults to AUTO
    Returns:
        A transformers pretrained model and tokenizer for usage within the framework"""

    return pretrained_from_info(model_info_from_name(model_name, model_src))
