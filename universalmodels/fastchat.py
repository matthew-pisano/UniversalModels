import subprocess
import time
from subprocess import Popen

from .constants import logger


class Worker:

    def __init__(self, model_path: str, port: int, worker_process: Popen, server_process: Popen):
        self.model_path = model_path
        self.port = port
        self.worker_process = worker_process
        self.server_process = server_process


class FastChatCtxMgr:
    """Allows fastchat to be used as a context manager"""

    def __init__(self, model_path: str):
        """

        Args:
            model_path: The model path for fastchat to open and close"""

        self.model_path = model_path

    def get_worker(self):
        return FastChatController.get_worker(self.model_path)

    def __enter__(self):
        FastChatController.open(self.model_path)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        FastChatController.close(self.model_path)


class FastChatController:
    """Manages fastchat servers and workers for quicker model inference"""

    _workers: dict[str, Worker] = {}
    controller_process: Popen | None = None
    _enabled = True

    port_generator = (i for i in range(8000, 8005))
    """Generates unique port numbers if multiple models are being used at once"""

    @classmethod
    def manager(cls, model_path: str):
        """Creates a fastchat context manager

        Args:
            model_path: The model path for the context manager to use
        Returns:
            A context manager"""

        return FastChatCtxMgr(model_path)

    @classmethod
    def is_active(cls):
        """Check is the fastchat controller is running

        Returns:
            Whether the fastchat controller is running"""

        return cls.controller_process is not None

    @classmethod
    def is_available(cls):
        """Check if the fastchat module is available and installed

        Returns:
            Whether the fastchat module is available"""

        p = Popen(['python3', '-c', 'import fastchat'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()
        return p.returncode == 0

    @classmethod
    def is_enabled(cls):
        """Checks if fastchat has been manually disabled within this program. It is enabled by default

        Returns:
            Whether fastchat is enabled"""

        return cls._enabled

    @classmethod
    def enable(cls):
        """Enables fastchat manually if it has been disabled"""

        cls._enabled = True

    @classmethod
    def disable(cls):
        """Manually disables fastchat"""

        cls._enabled = False

    @classmethod
    def get_worker(cls, model_path: str) -> Worker:
        """Gets a particular worker for a model

        Args:
            model_path: The model to get the worker for
        Returns:
            The worker associated with the given model"""

        if model_path not in cls._workers:
            raise ValueError(f"No worker found for model '{model_path}'")

        return cls._workers[model_path]

    @classmethod
    def open(cls, model_path: str, port: int = None):
        """Initiates the fastchat controller, server, and worker for a particular model

        Args:
            model_path: The model to use fastchat for
            port: The port to run the server on"""

        if not cls.is_available():
            raise ValueError('fastChat not available, please install fastchat to use it')
        if not cls.is_enabled():
            raise ValueError('fastChat has been disabled, please enable it to use it')

        # If there is already a fastchat worker for this model
        if model_path in cls._workers:
            return

        # Generate a port if none is provided
        if port is None:
            port = next(cls.port_generator)

        if cls.controller_process is None:
            logger.info("Initializing fastchat controller...")
            cls.controller_process = Popen(['python3', '-m', 'fastchat.serve.controller'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = cls.controller_process.communicate(timeout=3)
                raise RuntimeError(f"fastchat controller exited unexpectedly with code {cls.controller_process.returncode} and message {stderr.decode()}")
            except subprocess.TimeoutExpired:
                ...

        if model_path not in cls._workers:
            logger.info(f"Initializing {model_path} worker...")
            worker_process = Popen(['python3', '-m', 'fastchat.serve.model_worker', '--model-path', model_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = worker_process.communicate(timeout=10)
                raise RuntimeError(f"fastchat {model_path} worker exited unexpectedly with code {worker_process.returncode} and message {stderr.decode()}")
            except subprocess.TimeoutExpired:
                ...

            logger.info("Starting fastchat openai server...")
            server_process = Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', 'localhost', '--port', str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = server_process.communicate(timeout=15)
                raise RuntimeError(f"fastchat {model_path} server exited unexpectedly with code {server_process.returncode} and message {stderr.decode()}")
            except subprocess.TimeoutExpired:
                ...

            cls._workers[model_path] = Worker(model_path, port, worker_process, server_process)

        return cls._workers[model_path]

    @classmethod
    def close(cls, model_path: str = None):
        """Closes the fastchat processes for one or all models

        Args:
            model_path: The path of the model to close.  Closes all models if this is None"""

        if model_path is not None:
            logger.info(f"Closing {model_path} worker...")
            cls.get_worker(model_path).worker_process.kill()
            cls.get_worker(model_path).server_process.kill()
            cls._workers.pop(model_path)
        else:
            for path in list(cls._workers.keys()):
                cls.close(path)

            if cls.controller_process is not None:
                logger.info(f"Closing fastchat controller...")
                cls.controller_process.terminate()

                time.sleep(2)
                if cls.controller_process.poll() is None:
                    logger.warning("fastchat controller process was not terminated.  Forcing a kill")
                    cls.controller_process.kill()
                cls.controller_process = None
