from .supernet import SuperNetwork
from .hypernet import HyperNetwork

_MODEL_CLASSES = {
    "supernet": SuperNetwork,
    "hypernet": HyperNetwork,
}

def get_model_class(model_name: str):
    return _MODEL_CLASSES[model_name]
