import importlib


def get_dataset(alias):
    dataset_module = importlib.import_module('datasets.' + alias)
    return dataset_module.Dataset
