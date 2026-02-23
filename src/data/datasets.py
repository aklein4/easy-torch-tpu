
import datasets

from utils import constants


def get_dataset(url: str, kwargs: dict) -> datasets.Dataset:
    """
    Get a dataset by name.
    
    Args:
        url (str): The name of the dataset to retrieve.
        kwargs (dict): Additional arguments to pass to the dataset loader.
    
    Returns:
        datasets.Dataset: The requested dataset.
    """

    ds = datasets.load_dataset(url, **kwargs)

    if "streaming" in kwargs.keys() and kwargs["streaming"]:
        ds = ds.shard(num_shards=constants.PROCESS_COUNT(), index=constants.PROCESS_INDEX())

    return ds

