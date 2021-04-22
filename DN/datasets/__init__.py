from __future__ import absolute_import
import warnings

from .pitts import Pittsburgh
from .tokyo import Tokyo


__factory = {
    'pitts': Pittsburgh,
    'tokyo': Tokyo,
}

def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)

def create(name, root, *args, **kwargs):

    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)

def names():
    return sorted(__factory.keys())