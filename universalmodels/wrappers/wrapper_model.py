import time
from typing import Optional

import openai
import torch
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation.utils import GenerateOutput

from ..constants import logger
from ..mock_tokenizer import MockTokenizer


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
            resp = self.retry_generate(prompt, **kwargs)
            responses.append(tokenizer.encode(resp))

        return torch.LongTensor(responses)

    def retry_generate(self, prompt: str, retries=3, **kwargs) -> str:
        """Retries the generation of a text response from the given text prompt while an error is thrown

        Args:
            prompt: The plain text prompt
            retries: The number of retries to perform after an error before throwing an exception
        Returns:
            The generated response tokens"""

        if retries < 1:
            raise ValueError("Number of retries must be at least 1")

        # Linear backoff for retries
        backoff = 5
        # Loop until a response is successfully generated or the number of retries runs out
        while retries > 0:
            retries -= 1
            try:
                return self.generate_text(prompt, **kwargs)
            except (RuntimeError, openai.OpenAIError) as e:
                if retries <= 0:
                    raise e
                logger.warning(f"Model {self.name_or_path} received error {e}.\nRetrying ({retries} remaining)...")
                time.sleep(backoff)
                backoff += 2
