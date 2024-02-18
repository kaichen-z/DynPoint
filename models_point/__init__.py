import importlib

def get_model(alias, test=False):
    module = importlib.import_module('models_point.' + alias)
    return module.Model
