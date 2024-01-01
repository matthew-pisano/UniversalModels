

def test_import_all():
    try:
        import universalmodels
    except ImportError as e:
        raise AssertionError(str(e))


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
        from universalmodels import ModelSrc, model_info_from_name
    except ImportError as e:
        raise AssertionError(str(e))


def test_constants():
    try:
        from universalmodels.constants import set_seed, GLOBAL_SEED
    except ImportError as e:
        raise AssertionError(str(e))
