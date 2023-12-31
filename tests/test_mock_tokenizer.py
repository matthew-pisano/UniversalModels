import pytest

from universalmodels.mock_tokenizer import MockTokenizer


def is_int_list(int_list: list[int]):
    for int_value in int_list:
        if not isinstance(int_value, int):
            return False
    return True


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer("foo")


@pytest.fixture
def decoded_strings():
    return {
        "empty": "",
        "ascii": "hello \n\t\rthere",
        "unicode": "😀😀🤓🤓⟨б⟩, ⟨в⟩, ⟨г⟩, ⟨д⟩, ⟨ж⟩, ⟨з⟩, ⟨к⟩, ⟨л⟩, ⟨м⟩, ⟨н⟩, 水 (氵) 'water'  也 /*lAjʔ/  /*Cə.lraj/  drje  chí [ʈʂʰǐ]  ci4 [tsʰiː˩]  chi [tɕi] 馳  'gallop'  馬 'horse'  /*[l]raj/"
    }


def test_encode_empty(mock_tokenizer, decoded_strings):
    tokens = mock_tokenizer.encode(decoded_strings["empty"])
    assert is_int_list(tokens)


def test_encode_ascii(mock_tokenizer, decoded_strings):
    tokens = mock_tokenizer.encode(decoded_strings["ascii"])
    assert is_int_list(tokens)


def test_encode_unicode(mock_tokenizer, decoded_strings):
    tokens = mock_tokenizer.encode(decoded_strings["unicode"])
    assert is_int_list(tokens)


def test_decode_empty(mock_tokenizer, decoded_strings):
    tokens = mock_tokenizer.encode(decoded_strings["empty"])
    decoded = mock_tokenizer.decode(tokens)
    assert decoded == decoded_strings["empty"]


def test_decode_ascii(mock_tokenizer, decoded_strings):
    tokens = mock_tokenizer.encode(decoded_strings["ascii"])
    decoded = mock_tokenizer.decode(tokens)
    assert decoded == decoded_strings["ascii"]


def test_decode_unicode(mock_tokenizer, decoded_strings):
    tokens = mock_tokenizer.encode(decoded_strings["unicode"])
    decoded = mock_tokenizer.decode(tokens)
    assert decoded == decoded_strings["unicode"]
