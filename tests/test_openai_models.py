import io
import sys

import openai
import pytest
from torch import Tensor
from dotenv import load_dotenv

from universalmodels.mock_tokenizer import MockTokenizer
from universalmodels.wrappers.dev_model import DevModelEnum
from universalmodels.wrappers.openai_api_model import OpenAIAPIModel


load_dotenv()


@pytest.fixture
def prompt():
    return "Please respond with 'Hello'"


@pytest.fixture
def response():
    return "Hello"


def test_invalid_model():
    with pytest.raises(ValueError):
        OpenAIAPIModel(DevModelEnum.MULTILINE.value)


def test_unknown_model():
    with pytest.raises(ValueError):
        OpenAIAPIModel("openai/unknown")


def test_unknown_fastchat_model():
    with pytest.raises(ValueError):
        model = OpenAIAPIModel("fastchat/test")
        model.generate_text("Hello")


def test_inference(prompt, response):

    model = OpenAIAPIModel("openai/gpt-3.5-turbo")
    tokenizer = MockTokenizer("openai/gpt-3.5-turbo")
    tokens = tokenizer.encode(prompt)
    resp_tokens = model.generate(Tensor([tokens]).int())[0]
    decoded = tokenizer.decode(resp_tokens)
    assert response in decoded
