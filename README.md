# The Universal Model Adapter

This package acts as an adapter between [Huggingface Transformers](https://github.com/huggingface/transformers) and several different APIs.  As of now, these are the [Huggingface Inference API](https://huggingface.co/inference-api) and the [OpenAI Inference API](https://platform.openai.com/docs/api-reference).

This works by mock `transformers.PreTrainedModel` classes that share the same `generate()` method, but make API calls on the backend.  Several `dev` models are also available for mocking generation or performing debugging tasks.

## Use Case

This package is best used in projects that use multiple different model sources interchangeably.  In these kinds of projects, a unified generation interface greatly simplifies a lot of code.  For example, a project that uses text generated from both Huggingface models and GPT models from OpenAI's API.

## Quick Start

### Installing from Source

1. Clone repository

```bash
git clone https://github.com/matthew-pisano/UniversalModels
cd UniversalModels
```

2. Install package

```bash
pip3 install -e ".[fastchat]"
```

Installing the `fastchat` extra enables support for using fastchat on compatible locally installed huggingface models.  See [FastChat supported models](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md) for more information on which models are supported.

### Example Usage

```python
import torch
from universalmodels import set_seed, pretrained_from_name

# Set the global seed to encourage deterministic generation 
# NOTE: DOES NOT affect OpenAI API models
set_seed(42)

hf_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_model, hf_tokenizer = pretrained_from_name(hf_model_name)

hf_tokens = hf_tokenizer.encode("Repeat the following: 'Hello there from a huggingface model'")
hf_response = hf_model.generate(torch.Tensor([hf_tokens]).int())
print(hf_response)

oai_model_name = "openai/gpt-3.5"
oai_model, oai_tokenizer = pretrained_from_name(oai_model_name)

oai_tokens = oai_tokenizer.encode("Repeat the following: 'Hello there from an openai model'")
oai_response = oai_model.generate(torch.Tensor([oai_tokens]).int())
print(oai_response)
```

***NOTE:*** Make sure your API keys are set for OpenAI and Huggingface before using models that require them!