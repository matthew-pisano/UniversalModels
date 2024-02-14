import io
import sys

import pytest
from torch import Tensor

from universalmodels.mock_tokenizer import MockTokenizer
from universalmodels.wrappers.dev_model import DevModel


@pytest.fixture
def prompt():
    return "Continue: Hello there..."


@pytest.fixture
def response():
    return "General kenobi!"


def test_invalid_model():
    with pytest.raises(ValueError):
        DevModel("openai/gpt-3.5-turbo")


def test_unknown_model():
    with pytest.raises(ValueError):
        DevModel("dev/unknown")


def test_echo_model(prompt):

    model = DevModel("dev/echo")
    tokenizer = MockTokenizer("dev/echo")
    tokens = tokenizer.encode(prompt)
    resp_tokens = model.generate(Tensor([tokens]).int())[0]
    decoded = tokenizer.decode(resp_tokens)
    assert decoded == prompt


def test_human_model(prompt, response, monkeypatch):

    monkeypatch.setattr(sys, 'stdin', io.StringIO(response+"\n:q"))

    model = DevModel("dev/human")
    tokenizer = MockTokenizer("dev/human")
    tokens = tokenizer.encode(prompt)
    resp_tokens = model.generate(Tensor([tokens]).int())[0]
    decoded = tokenizer.decode(resp_tokens)
    assert decoded == response
