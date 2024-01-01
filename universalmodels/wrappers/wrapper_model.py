from typing import Optional

import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation.utils import GenerateOutput

from universalmodels.mock_tokenizer import MockTokenizer


class WrapperModel(PreTrainedModel):

    def __init__(self, model_name: str, **kwargs):
        """
        Args:
            model_name: The name of the model to use"""

        super().__init__(PretrainedConfig(name_or_path=model_name))
        self.name_or_path = model_name

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generates a text response from the given text prompt

        Args:
            prompt: The plain text prompt
        Returns:
            The plain text model response"""

        raise NotImplementedError()

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, **kwargs) -> GenerateOutput | torch.LongTensor:
        """Spoofs the pretrained model generation to make it fit for a wrapper's generation

        Args:
            inputs: The input tokens to use for generation
        Returns:
            The generated response tokens"""

        if len(inputs.shape) != 2:
            raise ValueError("Inputs must be 2D tensors of input token IDs (Ex. Tensor([[101, 98, ...], [...], ...]))")

        tokenizer = MockTokenizer(self.name_or_path)
        responses = []

        for encoded_prompt in inputs:
            prompt = tokenizer.decode(encoded_prompt.tolist())
            resp = self.generate_text(prompt, **kwargs)
            responses.append(tokenizer.encode(resp))

        return torch.LongTensor(responses)
