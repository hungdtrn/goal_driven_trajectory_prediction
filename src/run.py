from attrdict import AttrDict

from .models.gtp import runner as gtp

def get_runner(model_name):
    if model_name == "gtp":
        return gtp