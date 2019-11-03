"""
Some global that we'll use
"""
import torch
from torchvision import models
from torchvision import transforms
from project.utils.cli import eprint

_LARGE = 512
_SMALL = 128


def _imsize(use_large=False):
    if use_large:
        eprint("Using image size of {0} instead of {1}".format(_LARGE, _SMALL))
        return _LARGE
    has_gpu = torch.cuda.is_available()
    return _LARGE if has_gpu else _SMALL


class NNSetup:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    IMSIZE = _imsize()
    LOADER = transforms.Compose(
        [transforms.Resize(IMSIZE), transforms.ToTensor()])
    UNLOADER = transforms.ToPILImage()  # reconvert into PIL image
    LOADER = None
    CNN = models.vgg19(
        pretrained=True).features.to(DEVICE).eval()
    CNN_NORMALIZATION_MEAN = torch.FloatTensor(
        [0.485, 0.456, 0.406]).to(DEVICE)
    CNN_NORMALIZATION_STD = torch.FloatTensor(
        [0.229, 0.224, 0.225]).to(DEVICE)
    CONTENT_LAYERS_DEFAULT = ["conv_4"]
    STYLE_LAYERS_DEFAULT = [
        "conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    @staticmethod
    def init(use_large=False):
        NNSetup.IMSIZE = _imsize(use_large=use_large)
        NNSetup.LOADER = transforms.Compose(
            [transforms.Resize(NNSetup.IMSIZE), transforms.ToTensor()])
