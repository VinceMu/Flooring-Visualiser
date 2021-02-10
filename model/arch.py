from enum import Enum, unique, auto
from mit_semseg.models import ModelBuilder


@unique
class EncoderArch(Enum):
    mobilenetv2dilated = auto()
    resnet101dilated = auto()
    resnet50dilated = auto()
    resnet18dilated = auto()
    resnet50 = auto()
    resnet101 = auto()
    hrnetv2 = auto()


@unique
class DecoderArch(str, Enum):
    ppm_deepsup = auto()
    c1_deepsup = auto()
    uper_net = auto()
    c1 = auto()


class EncoderParams:
    def __init__(self, arch="", fc_dim=-1, weights=""):
        self.arch = arch
        self.fc_dim = fc_dim
        self.weights = weights

    def getEncoder(self):
        return ModelBuilder.build_encoder(**self.__dict__)


class DecoderParams:
    def __init__(self,
                 arch="",
                 fc_dim=-1,
                 weights="",
                 num_class=150,
                 use_softmax=True):
        self.arch = arch
        self.fc_dim = fc_dim
        self.weights = weights
        self.num_class = num_class
        self.use_softmax = use_softmax

    def getDecoder(self):
        return ModelBuilder.build_decoder(**self.__dict__)
