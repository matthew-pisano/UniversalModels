import os
import logging
import sys

import transformers

logger = logging.getLogger("UniversalModels")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('[%(levelname)s %(name)s @ %(asctime)s] %(message)s', "%H:%M:%S"))
logger.addHandler(console_handler)

GLOBAL_SEED = None


def set_seed(seed: int):
    """Sets the global random seed for keeping inference as deterministic as possible

    Args:
        seed: The seed to set"""

    global GLOBAL_SEED
    logger.info(f"Setting random seed to {seed}")
    GLOBAL_SEED = seed
    transformers.set_seed(GLOBAL_SEED)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
