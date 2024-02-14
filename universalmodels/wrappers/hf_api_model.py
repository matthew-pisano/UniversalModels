import json
import os
from typing import Optional

import torch
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from transformers.generation.utils import GenerateOutput

from .wrapper_model import WrapperModel
from ..constants import logger


class HFAPIModel(WrapperModel):
    """Huggingface API Model wrapper.  Spoofs pretrained model generation while really generating text through the Huggingface API"""

    def __init__(self, model_name: str, model_task: str, **kwargs):
        """
        Args:
            model_name: The name of the huggingface model to use
            model_task: The name of the huggingface model task to perform"""

        try:
            # Check that the model can be loaded by huggingface
            AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise e

        super().__init__(model_name, **kwargs)
        self.model_task = model_task

    def generate_text(self, prompt: str, do_sample=True, temperature=0.7, max_new_tokens=None, timeout=10, **kwargs):
        """Generates a text response from the given text prompt
        
        Args:
            prompt: The plain text prompt
            do_sample: Whether to use the sampling decoding method
            temperature: The temperature of the model
            max_new_tokens: The maximum number of new tokens to generate
            timeout: The timeout for API requests
        Returns:
            The plain text huggingface model response"""

        # Set default params depending on the huggingface task to perform
        hf_params = {}
        if self.model_task == "conversational":
            hf_params = {"max_new_tokens": max_new_tokens, "return_full_text": False, "repetition_penalty": 1.5, "do_sample": do_sample, "temperature": temperature}
        elif self.model_task == "summarization":
            hf_params = {"max_new_tokens": 250}

        char_limit = 450
        if len(prompt) > char_limit:
            logger.warning(f"Prompt given to Huggingface API is too long! {len(prompt)} > {char_limit}.  This prompt will be truncated.")
            prompt = prompt[:char_limit // 2] + prompt[-char_limit // 2:]

        inference_client = InferenceClient(model=self.name_or_path, token=os.environ.get("HF_API_KEY"), timeout=timeout)
        resp = json.loads(inference_client.post(json={"inputs": prompt, "parameters": hf_params}).decode())

        if 'error' in resp:
            raise RuntimeError(f"Received error from HuggingFace API: {resp['error']}")

        return resp[0]["generated_text"]

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, do_sample=True, temperature=0.7,
                 max_new_tokens=None, timeout=30, retries=2, **kwargs) -> GenerateOutput | torch.LongTensor:
        """Spoofs the pretrained model generation to make it fit for huggingface API generation

        Args:
            inputs: The input tokens to use for generation
            do_sample: Whether to use the sampling decoding method
            temperature: The temperature of the model
            max_new_tokens: The maximum number of new tokens to generate
            timeout: The timeout for API requests
            retries: The number of retries to perform after an API error before throwing an exception
        Returns:
            The generated response tokens"""

        return super().generate(inputs, do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, timeout=timeout, retries=retries, **kwargs)
