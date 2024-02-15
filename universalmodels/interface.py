from enum import Enum

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoModel

from .constants import logger
from .fastchat import FastChatController
from .mock_model import MockModel
from .mock_tokenizer import MockTokenizer
from .wrappers.dev_model import DevModel
from .wrappers.hf_api_model import HFAPIModel
from .wrappers.openai_api_model import OpenAIAPIModel


class ModelSrc(Enum):
    """Valid sources for loading and running models"""

    AUTO = "auto"
    """Placeholder value to automatically decide the model source"""

    NO_LOAD = "no_load"
    """Does not load a model.  Used for evaluation when a model is specified, but no inference is done"""

    HF_LOCAL = "huggingface_local"
    """Models are downloaded locally and run on a GPU"""

    HF_API = "huggingface_hub"
    """Models are run in the cloud through the Huggingface API"""

    OPENAI_API = "openai"
    """Models are run in the cloud through the OpenAI API or run locally using fastchat"""

    DEV = "dev"
    """Models are run locally using manual input or predetermined algorithms.  Used for testing and development purposes"""


class ModelInfo:

    def __init__(self, pretrained_model_name_or_path: str, model_src: ModelSrc,
                 model_class: PreTrainedModel | None = AutoModelForCausalLM,
                 tokenizer_class: PreTrainedTokenizer | None = AutoTokenizer, model_task: str = None, fp_precision: int = 16):
        """
        Args:
            pretrained_model_name_or_path: The name of the underlying model to use
            model_src: The suggested source of the model to load. Defaults to AUTO
            model_class: The class of transformers PreTrainedModel to use
            tokenizer_class: The class of transformers PreTrainedTokenizer to use
            model_task: The huggingface task for the model to perform, if applicable
            fp_precision: The precision of the model's calculations.  4-bit and 8-bit precision use quantization"""

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_src = model_src
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.model_task = model_task

        if fp_precision not in [32, 16, 8, 4]:
            raise ValueError("fp_precision must be one of 32, 16, 8, 4")
        self._fp_precision = fp_precision

        if self.model_src == ModelSrc.HF_API and self.model_task is None:
            raise ValueError("A model task is required to use HuggingFace models")

    @property
    def fp_precision(self):
        return self._fp_precision

    def as_dict(self):
        return vars(self)

    def __eq__(self, other):
        if isinstance(other, ModelInfo):
            return self.as_dict() == other.as_dict()
        else:
            return False


def model_info_from_name(target_model_name: str, model_src: ModelSrc = ModelSrc.AUTO, model_task: str = None) -> ModelInfo:
    """Gets information for creating a framework model from the name of the underlying model. Agnostic of which framework model this is being used for

    Args:
        target_model_name: The name of the underlying model to use
        model_src: The suggested source of the model to load. Defaults to AUTO
        model_task: The name of the task a huggingface API model should perform, if applicable. Defaults to None
    Returns:
        A ModelInfo instance containing the information necessary to find the given model"""

    if model_src == ModelSrc.NO_LOAD:
        return ModelInfo(target_model_name, ModelSrc.NO_LOAD, None, None, model_task=model_task)

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
            return ModelInfo(target_model_name, ModelSrc.HF_API, None, None, model_task=model_task)
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
        case _:
            raise ValueError(f"Invalid model source {model_src}")


def pretrained_from_info(model_info: ModelInfo) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Gets the pretrained model and tokenizer from the given model information

    Args:
        model_info: The model and tokenizer information to get the pretrained model and tokenizer
    Returns:
        A transformers pretrained model and tokenizer for usage within the framework"""

    # Appropriately assign any AUTO model info
    if model_info.model_src == ModelSrc.AUTO:
        model_info = model_info_from_name(model_info.pretrained_model_name_or_path)

    logger.debug(f"Loading a pretrained model {model_info.pretrained_model_name_or_path} from {model_info.model_src}")

    match model_info.model_src:
        case ModelSrc.NO_LOAD:
            return MockModel(model_info.pretrained_model_name_or_path, model_info.model_task), MockTokenizer(model_info.pretrained_model_name_or_path)
        case ModelSrc.OPENAI_API:
            return OpenAIAPIModel(model_info.pretrained_model_name_or_path), MockTokenizer(model_info.pretrained_model_name_or_path)
        case ModelSrc.HF_API:
            return HFAPIModel(model_info.pretrained_model_name_or_path, model_info.model_task), MockTokenizer(model_info.pretrained_model_name_or_path)
        case ModelSrc.DEV:
            return DevModel(model_info.pretrained_model_name_or_path), MockTokenizer(model_info.pretrained_model_name_or_path)
        case ModelSrc.HF_LOCAL:
            fp_kwargs = {}
            match model_info.fp_precision:
                case 32:
                    fp_kwargs["torch_dtype"] = torch.float32
                case 16:
                    fp_kwargs["torch_dtype"] = torch.bfloat16
                case 8:
                    fp_kwargs["load_in_8bit"] = True
                case 4:
                    fp_kwargs["load_in_4bit"] = True
                case _:
                    raise ValueError("fp_precision must be one of 32, 16, 8, 4")

            try:
                model = model_info.model_class.from_pretrained(model_info.pretrained_model_name_or_path, **fp_kwargs)
            except ValueError as e:
                logger.warning(f"Could not load {model_info.pretrained_model_name_or_path} as a {model_info.model_class} model.  Using AutoModel instead.")
                model = AutoModel.from_pretrained(model_info.pretrained_model_name_or_path, **fp_kwargs)

            return model, model_info.tokenizer_class.from_pretrained(model_info.pretrained_model_name_or_path)
        case _:
            raise ValueError(f"Invalid model source {model_info.model_src}")


def pretrained_from_name(model_name: str, model_src: ModelSrc = ModelSrc.AUTO, model_task: str = None):
    """Gets the pretrained model and tokenizer from the given model name

    Args:
        model_name: The name of the underlying model to use
        model_src: The suggested source of the model to load. Defaults to AUTO
        model_task: The huggingface task for the model to perform, if applicable
    Returns:
        A transformers pretrained model and tokenizer for usage within the framework"""

    return pretrained_from_info(model_info_from_name(model_name, model_src=model_src, model_task=model_task))
