import pytest
import torch
from transformers import T5ForConditionalGeneration

from universalmodels import pretrained_from_info, model_info_from_name, ModelInfo


@pytest.fixture
def inference_prompt():
    return "Complete the following quote: You were the chosen one! It was said that you would destroy the Sith, not join them! Bring balance to the Force, not"


@pytest.fixture
def hf_model_info():
    model_name = "google/flan-t5-small"
    model_info = model_info_from_name(model_name)
    model_info.model_class = T5ForConditionalGeneration
    return model_info


def hf_local_inference(model_info: ModelInfo, prompt):

    model, tokenizer = pretrained_from_info(model_info)
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)])
    resp_tokens = model.generate(prompt_tokens)[0]
    return tokenizer.decode(resp_tokens, skip_special_tokens=True)


def test_hf_local_bad_precision_inference(hf_model_info, inference_prompt):

    hf_model_info._fp_precision = 42
    with pytest.raises(ValueError):
        hf_local_inference(hf_model_info, inference_prompt)


def test_hf_local_fp32_inference(hf_model_info, inference_prompt, request):

    hf_model_info._fp_precision = 32
    response = hf_local_inference(hf_model_info, inference_prompt)
    print(f"{request.node.name} response:", response)


def test_hf_local_fp16_inference(hf_model_info, inference_prompt, request):

    hf_model_info._fp_precision = 16
    response = hf_local_inference(hf_model_info, inference_prompt)
    print(f"{request.node.name} response:", response)


def test_hf_local_fp8_inference(hf_model_info, inference_prompt, request):

    hf_model_info._fp_precision = 8
    response = hf_local_inference(hf_model_info, inference_prompt)
    print(f"{request.node.name} response:", response)


def test_hf_local_fp4_inference(hf_model_info, inference_prompt, request):

    hf_model_info._fp_precision = 4
    response = hf_local_inference(hf_model_info, inference_prompt)
    print(f"{request.node.name} response:", response)
