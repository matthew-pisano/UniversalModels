import time
from typing import Optional

import torch
import openai
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation.utils import GenerateOutput

from .wrapper_model import WrapperModel
from ..logger import root_logger
from ..mock_tokenizer import MockTokenizer
from ..constants import GLOBAL_SEED
from ..fastchat import FastChatController


class OpenAIAPIModel(WrapperModel):
    """OpenAI API Model wrapper.  Spoofs pretrained model generation while really generating text through the OPENAI API"""

    def __init__(self, model_name: str, **kwargs):
        """
        Args:
            model_name: The name of the OpenAI model to use"""

        super().__init__(model_name, **kwargs)

    def generate_text(self, prompt: str, timeout=10, retries=2, **kwargs):
        """Generates a text response from the given text prompt

        Args:
            prompt: The plain text prompt
            timeout: The timeout for API requests
            retries: The number of retries to perform after an API error before throwing an exception
        Returns:
            The plain text OpenAI model response"""

        response_str = None

        # Generation from the normal OpenAI API
        if self.name_or_path.startswith("openai/"):

            # Loop until a response is successfully generated from the API or the number of retries runs out
            while retries > 0:
                retries -= 1
                try:
                    openai.api_base = "https://api.openai.com/v1"
                    resp = openai.ChatCompletion.create(model=self.name_or_path.split("/")[-1], messages=[
                        {"role": "user", "content": prompt}], seed=GLOBAL_SEED, request_timeout=timeout, **kwargs)
                    response_str = resp["choices"][0]["message"]["content"]
                    break
                except Exception as e:
                    if retries <= 0:
                        raise e
                    root_logger.warning(f"Received error {e} from OpenAI API.  Retrying...")
                    time.sleep(5)
        # Generation from the fastchat API
        else:
            openai.api_base = f"http://localhost:{FastChatController.get_worker(self.name_or_path)['port']}/v1"
            resp = openai.Completion.create(model=self.name_or_path.split("/")[-1], prompt=prompt, **kwargs)
            response_str = resp["choices"][0]["text"]

        if response_str is None:
            raise ValueError("Response encoding has not been properly generated")

        return response_str

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, timeout=10, retries=2, **kwargs) -> GenerateOutput | torch.LongTensor:
        """Spoofs the pretrained model generation to make it fit for OpenAI API generation

        Args:
            inputs: The input tokens to use for generation
            timeout: The timeout for API requests
            retries: The number of retries to perform after an API error before throwing an exception
        Returns:
            The generated response tokens"""

        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs["max_new_tokens"]
            kwargs.pop("max_new_tokens")

        if "do_sample" in kwargs:
            kwargs.pop("do_sample")

        return super().generate(inputs, timeout=timeout, retries=retries, **kwargs)
