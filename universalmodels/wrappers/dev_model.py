from enum import Enum
from typing import Optional

import torch
from transformers.generation.utils import GenerateOutput

from .wrapper_model import WrapperModel
from ..logger import root_logger


class DevModelEnum(Enum):
    """An enum of the valid dev models"""

    HUMAN = "dev/human"
    ECHO = "dev/echo"


class DevModel(WrapperModel):
    """Developer Model wrapper.  Spoofs pretrained model generation while really generating text through manual input or predetermined methods"""

    def __init__(self, model_name: str, **kwargs):
        """
        Args:
            model_name: The name of the developer model to use"""

        if not model_name.startswith("dev"):
            raise ValueError("Dev models must have names in the form of 'dev/*'")
        if model_name not in [model.value for model in DevModelEnum]:
            raise ValueError(f"Unknown dev model '{model_name}'")

        super().__init__(model_name, **kwargs)

    def generate_text(self, prompt: str,  **kwargs):

        match self.name_or_path:
            case DevModelEnum.HUMAN.value:
                response_str = self._generate_manual(prompt)
            case DevModelEnum.ECHO.value:
                response_str = self._generate_echo(prompt)
            case _:
                raise ValueError(f"Could not find dev model with name '{self.name_or_path}'")

        return response_str

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, **kwargs) -> GenerateOutput | torch.LongTensor:
        """Spoofs the pretrained model generation to make it fit for custom development generation

        Args:
            inputs: The input tokens to use for generation
        Returns:
            The generated response tokens"""

        return super().generate(inputs, **kwargs)

    @staticmethod
    def _generate_manual(prompt: str):
        """Allows for users to generate responses to the prompt themselves through standard input for debugging purposes

        Args:
            prompt: The prompt to show to standard output
        Returns:
            The manually generated response"""

        root_logger.unchecked("[MANUAL PROMPT]\n", prompt)
        root_logger.info("[MANUAL INSTRUCTIONS] Enter ':q' on a new line submit your response and to quit")

        resp = ""
        while True:
            partial_resp = input(">>> ")
            if partial_resp == ":q":
                break
            elif partial_resp.startswith(":") and len(partial_resp) == 2:
                raise ValueError(f"Unrecognized command '{partial_resp}'")

            resp += partial_resp+"\n"

        return resp.rstrip("\n")

    @staticmethod
    def _generate_echo(prompt: str):
        """Simply echoes the prompt

        Args:
            prompt: The prompt to clone as the response

        Returns:
            The unchanged prompt itself as the response"""

        return prompt
