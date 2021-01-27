from enum import Enum, unique

@unique
class EncoderArch(str, Enum):
    MOBILE_NET_V2_DILATED = 'mobilenetv2dilated'
    RESNET_101_DILATED = 'resnet101dilated'
    RESNET_50_DILATED = 'resnet50dilated'


@unique
class DecoderArch(str, Enum):
    PPM_DEEPSUP = 'ppm_deepsup'
    C1_DEEPSUP = 'c1_deepsup'
