import pytest


def test_import_dev_wrapper():
    try:
        from universalmodels import DevModel
    except ImportError as e:
        raise AssertionError(str(e))


def test_import_hf_wrapper():
    try:
        from universalmodels import HFAPIModel
    except ImportError as e:
        raise AssertionError(str(e))


def test_import_openai_wrapper():
    try:
        from universalmodels import OpenAIAPIModel
    except ImportError as e:
        raise AssertionError(str(e))


def test_import_fastchat_controller():
    try:
        from universalmodels.fastchat import FastChatController
    except ImportError as e:
        raise AssertionError(str(e))


def test_import_interface():
    try:
        from universalmodels import set_seed, ModelSrc, model_info_from_name, GLOBAL_SEED
    except ImportError as e:
        raise AssertionError(str(e))
