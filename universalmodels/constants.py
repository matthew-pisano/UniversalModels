import os

import transformers


GLOBAL_SEED = None


def set_seed(seed: int):
    """Sets the global random seed for keeping inference as deterministic as possible

    Args:
        seed: The seed to set"""

    global GLOBAL_SEED
    print(f"Setting random seed to {seed}")
    GLOBAL_SEED = seed
    transformers.set_seed(GLOBAL_SEED)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
