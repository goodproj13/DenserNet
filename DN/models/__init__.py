from __future__ import absolute_import

from .vgg import *
from .netvlad import *


__factory = {
    'vgg16': vgg16,
    'netvlad': NetVLAD,
    'embednet': EmbedNet,
    'embednetpca': EmbedNetPCA,
    'embedregionnet': EmbedRegionNet,
}


def create(name, *args, **kwargs):

    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)


def names():
    return sorted(__factory.keys())
