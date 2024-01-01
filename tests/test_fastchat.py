import time

import pytest
from dotenv import load_dotenv

from universalmodels import model_info_from_name, ModelSrc
from universalmodels.fastchat import FastChatController


pytestmark = pytest.mark.skipif(not FastChatController.is_available(), reason="FastChat is not installed or is not available")
load_dotenv()


@pytest.fixture
def model_path():
    return "meta-llama/Llama-2-7b-chat-hf"


def check_open(model_path):

    assert FastChatController.controller_process.poll() is None
    assert model_path in FastChatController._workers

    worker = FastChatController.get_worker(model_path)
    assert worker.worker_process.poll() is None
    assert worker.server_process.poll() is None


def check_close_all(controller_process, workers):

    assert controller_process.poll() is not None
    assert FastChatController.controller_process is None

    for worker in workers.values():
        assert worker.worker_process.poll() is not None
        assert worker.server_process.poll() is not None

    assert len(FastChatController._workers) == 0


def test_open(model_path):

    FastChatController.close()
    FastChatController.open(model_path)
    check_open(model_path)


def test_close_all(model_path):

    if not FastChatController.is_active():
        FastChatController.open(model_path)

    controller_process = FastChatController.controller_process
    workers = {**FastChatController._workers}
    FastChatController.close()

    time.sleep(2)
    check_close_all(controller_process, workers)


def test_ctx_mgr(model_path):

    FastChatController.close()
    with FastChatController.manager(model_path) as manager:
        check_open(model_path)
        assert manager.get_worker().model_path == model_path
        worker = FastChatController.get_worker(model_path)

    time.sleep(2)
    assert worker.server_process.poll() is not None
    assert worker.worker_process.poll() is not None
    assert model_path not in FastChatController._workers


def test_disable():

    FastChatController.disable()
    with pytest.raises(RuntimeError):
        model_info_from_name("meta-llama/Llama-2-7b-chat-hf", ModelSrc.OPENAI_API)
