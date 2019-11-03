"""
Library for running this
TODO:
    probably should abstract saving the image to the main?
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.utils import save_image

from project.art_style.style_transfer import run_style_transfer
from project.utils.args import CopyStyleArgs
from project.utils.utils import NNSetup


def load_image(image):
    # fake batch dimension required to fit network's input dimensions
    image = NNSetup.LOADER(image).unsqueeze(0)
    # pylint: disable=E1101
    return image.to(NNSetup.DEVICE, torch.float)
    # pylint: enable=E1101


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = NNSetup.UNLOADER(image)
    if title is not None:
        plt.title(title)
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run(args: CopyStyleArgs):
    NNSetup.init(use_large=args.max_size)
    if args.progress_dir:
        with open(Path(args.progress_dir, "args.txt"), 'w') as arg_file:
            arg_file.write(str(args))

    style_img: Image = Image.open(args.style_image)
    content_img: Image = Image.open(args.image_path)
    if args.resize and style_img.size != content_img.size:
        if style_img.size > content_img.size:
            content_img = content_img.resize(style_img.size)
        else:
            style_img = style_img.resize(content_img.size)
    style_img = load_image(style_img)
    content_img = load_image(content_img)
    assert (
        style_img.size() == content_img.size()
    ), "we need to import style and content images of the same size"

    plt.figure()
    imshow(style_img, title="Style Image")
    if args.progress_dir:
        plt.savefig(Path(args.progress_dir, "style-image.png"))

    plt.figure()
    imshow(content_img, title="Content Image")
    if args.progress_dir:
        plt.savefig(Path(args.progress_dir, "content-image.png"))
    input_img = content_img.clone()

    plt.figure()
    imshow(input_img, title="Input Image")
    if args.progress_dir:
        plt.savefig(Path(args.progress_dir, "input-image.png"))
    output = run_style_transfer(
        NNSetup.CNN,
        NNSetup.CNN_NORMALIZATION_MEAN,
        NNSetup.CNN_NORMALIZATION_STD,
        content_img,
        style_img,
        input_img,
        progress_dir=args.progress_dir,
    )

    plt.figure()
    imshow(output, title="Output Image")
    if args.progress_dir:
        plt.savefig(Path(args.progress_dir, "output-image.png"))

    save_image(output, args.output_name)
