

def test_import_all():
    import universalmodels


def test_import_dev_wrapper():
    from universalmodels import DevModel


def test_import_hf_wrapper():
    from universalmodels import HFAPIModel


def test_import_openai_wrapper():
    from universalmodels import OpenAIAPIModel


def test_import_fastchat_controller():
    from universalmodels.fastchat import FastChatController


def test_import_interface():
    from universalmodels import ModelSrc, model_info_from_name


def test_constants():
    from universalmodels.constants import set_seed, GLOBAL_SEED
