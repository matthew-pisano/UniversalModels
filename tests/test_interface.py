from dotenv import load_dotenv

from universalmodels import ModelSrc, ModelInfo, model_info_from_name, pretrained_from_name
from universalmodels.mock_tokenizer import MockTokenizer
from universalmodels.wrappers.hf_api_model import HFAPIModel
from universalmodels.wrappers.openai_api_model import OpenAIAPIModel

load_dotenv()


def test_model_info_from_name_hf():
    model_name = "google/flan-t5-small"
    model_info = model_info_from_name(model_name)
    assert model_info.model_src == ModelSrc.HF_LOCAL


def test_model_info_from_name_oai():
    model_name = "openai/gpt.3.5"
    model_info = model_info_from_name(model_name)
    assert model_info == ModelInfo(model_name, ModelSrc.OPENAI_API, None, None)


def test_model_info_from_name_fastchat():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_info = model_info_from_name(model_name, ModelSrc.OPENAI_API)
    assert model_info == ModelInfo(model_name, ModelSrc.OPENAI_API, None, None)


def test_openai_pretrained_from_name():
    model_name = "openai/gpt-3.5"
    model, tokenizer = pretrained_from_name(model_name)

    assert isinstance(model, OpenAIAPIModel)
    assert isinstance(tokenizer, MockTokenizer)


def test_hf_pretrained_from_name():
    model_name = "google/flan-t5-small"
    model, tokenizer = pretrained_from_name(model_name)

    assert isinstance(model, HFAPIModel)
    assert isinstance(tokenizer, MockTokenizer)
