from .AVE import *
from .VGG_subset import *

dataset_mapping = {
    'AVE': AVEBuilder,
    'VGG_subset': VGGSubsetBuilder
}