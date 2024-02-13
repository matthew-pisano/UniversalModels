import io
import sys

import pytest
from torch import Tensor
from dotenv import load_dotenv

from universalmodels.mock_tokenizer import MockTokenizer
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
        OpenAIAPIModel("dev/human")


def test_unknown_model():
    with pytest.raises(ValueError):
        OpenAIAPIModel("openai/unknown")


def test_gpt_3_5_model(prompt, response):

    model = OpenAIAPIModel("openai/gpt-3.5-turbo")
    tokenizer = MockTokenizer("openai/gpt-3.5-turbo")
    tokens = tokenizer.encode(prompt)
    resp_tokens = model.generate(Tensor([tokens]).int())[0]
    decoded = tokenizer.decode(resp_tokens)
    assert response in decoded
