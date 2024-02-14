from typing import Optional

import torch
from openai import OpenAI

from transformers.generation.utils import GenerateOutput

from .wrapper_model import WrapperModel
from ..constants import GLOBAL_SEED
from ..fastchat import FastChatController


class OpenAIAPIModel(WrapperModel):
    """OpenAI API Model wrapper.  Spoofs pretrained model generation while really generating text through the OPENAI API"""

    def __init__(self, model_name: str, **kwargs):
        """
        Args:
            model_name: The name of the OpenAI model to use"""

        if model_name.startswith("openai/"):
            client = OpenAI()
            valid_models = [model.id for model in client.models.list()]
            if model_name.replace("openai/", "") not in valid_models:
                raise ValueError(f"Unknown OpenAI model '{model_name}'")
        if model_name.startswith("dev/"):
            raise ValueError("OpenAI models must not have names in the form of 'dev/*'")

        super().__init__(model_name, **kwargs)

    def generate_text(self, prompt: str, timeout=10, **kwargs):
        """Generates a text response from the given text prompt

        Args:
            prompt: The plain text prompt
            timeout: The timeout for API requests
        Returns:
            The plain text OpenAI model response"""

        if self.name_or_path.startswith("openai/"):
            # Generation from the normal OpenAI API
            client = OpenAI()
            resp = client.chat.completions.create(model=self.name_or_path.split("/")[-1], messages=[
                {"role": "user", "content": prompt}], seed=GLOBAL_SEED, timeout=timeout, **kwargs)
            return resp.choices[0].message.content
        else:
            # Generation from the fastchat API
            client = OpenAI(base_url=f"http://localhost:{FastChatController.get_worker(self.name_or_path).port}/v1")
            resp = client.completions.create(model=self.name_or_path.split("/")[-1], prompt=prompt, **kwargs)
            return resp.choices[0].text

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
