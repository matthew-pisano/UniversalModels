# The Universal Model Adapter

This package acts as an adapter between [Huggingface Transformers](https://github.com/huggingface/transformers) and several different APIs.  As of now, these are the [Huggingface Inference API](https://huggingface.co/inference-api) and the [OpenAI Inference API](https://platform.openai.com/docs/api-reference).

This works by mock `transformers.PreTrainedModel` classes that share the same `generate()` method, but make API calls on the backend.  Several `dev` models are also available for mocking generation or performing debugging tasks.

## Use Case

This package is best used in projects that use multiple different model sources interchangeably.  In these kinds of projects, a unified generation interface greatly simplifies a lot of code.  For example, a project that uses text generated from both Huggingface models and GPT models from OpenAI's API.

### Fine-Gained Source Control

An advantage of this package is that it can either automatically resolve the source of a model from its name, or you can specify the source (OpenAI, Huggingface, etc.) manually.  This can be done through an extra parameter to the `pretrained_from_...()` methods.  For example:

```python
from universalmodels.interface import pretrained_from_name, ModelSrc

model_name = "mistralai/Mistral-7B-v0.1"
# This will automatically resolve the model's source to 
# a local Huggingface transformers model (ModelSrc.HF_LOCAL)
local_model, tokenizer = pretrained_from_name(model_name, model_src=ModelSrc.AUTO)

# This will attempt to start the FastChat service and run 
# a local instance of the OpenAI API to run optimized generation
fschat_model, tokenizer = pretrained_from_name(model_name, model_src=ModelSrc.OPENAI_API)

# This will create a mock model without any generation logic attached.
# This is useful for when the shell of a model is needed as a reference.
# This option does not load any local models into memory or activate FastChat.
mock_model, tokenizer = pretrained_from_name(model_name, model_src=ModelSrc.NO_LOAD)
```

## Quick Start

### Installing from PyPI

```bash
pip3 install "universalmodels[fastchat]"
```

### Installing from Source

```bash
git clone https://github.com/matthew-pisano/UniversalModels
cd UniversalModels
pip3 install -e ".[fastchat]"
```

Installing the `fastchat` extra enables support for using fastchat on compatible locally installed huggingface models.  See [FastChat supported models](https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md) for more information on which models are supported.

### Example Usage

In the following example, note that the interfaces for the Huggingface and OpenAI modles are the same.  This is the primary benefit of using this package.

```python
import torch
from universalmodels import pretrained_from_name
from universalmodels.constants import set_seed

# Set the global seed to encourage deterministic generation 
# NOTE: DOES NOT affect OpenAI API models
set_seed(42)

# Huggingface model example
hf_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_model, hf_tokenizer = pretrained_from_name(hf_model_name)

hf_tokens = hf_tokenizer.encode("Repeat the following: 'Hello there from a huggingface model'")
hf_resp_tokens = hf_model.generate(torch.Tensor([hf_tokens]).int())[0]
hf_response = hf_tokenizer.decode(hf_resp_tokens)
print(hf_response)

# OpenAI model example
oai_model_name = "openai/gpt-3.5"
oai_model, oai_tokenizer = pretrained_from_name(oai_model_name)

oai_tokens = oai_tokenizer.encode("Repeat the following: 'Hello there from an openai model'")
oai_resp_tokens = oai_model.generate(torch.Tensor([oai_tokens]).int())[0]
oai_response = oai_tokenizer.decode(oai_resp_tokens)
print(oai_response)
```

> [!IMPORTANT]
> Make sure your API keys are set for OpenAI and Huggingface before using models that require them!
