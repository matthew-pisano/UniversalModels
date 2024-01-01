from typing import Union, List

from transformers import PreTrainedTokenizer, AddedToken
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput


class MockTokenizer(PreTrainedTokenizer):
    """A mock tokenizer to translate strings to character integer ids in a lossless fashion"""

    def __init__(self, tokenizer_name: str, **kwargs):
        super().__init__()
        self.tokenizer_name = tokenizer_name

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        ...

    def vocab_size(self) -> int:
        return 50257

    @staticmethod
    def encode(text: TextInput | PreTokenizedInput | EncodedInput, **kwargs):
        """Spoofs the pretrained tokenizer's encoding by converting characters to integers"""

        if type(text) is str:
            text = [text]

        encoded = []
        for seq in text:
            for char in seq:
                encoded.append(ord(char))

        return encoded

    @staticmethod
    def decode(token_ids: int | list[int], **kwargs):
        """Spoofs the pretrained tokenizer's decoding by converting integers to characters"""

        if type(token_ids) is int:
            token_ids = [token_ids]

        decoded = ""
        for char_id in token_ids:
            decoded += chr(int(char_id))

        return decoded

    def __repr__(self):

        return f"{self.__class__.__name__}(tokenizer_name='{self.tokenizer_name}')"
