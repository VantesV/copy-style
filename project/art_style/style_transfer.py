from project.art_style.model import get_style_model_and_losses
from project.art_style.optimizer import get_input_optimizer
from project.utils.cli import eprint
from torchvision.utils import save_image
from torch import FloatTensor
from pathlib import Path


def run_style_transfer(
    cnn,
    normalization_mean,
    normalization_std,
    content_img,
    style_img,
    input_img,
    num_steps=300,
    style_weight=1000000,
    content_weight=1,
    progress_dir: Path = None,
) -> FloatTensor:
    """Run the style transfer."""
    eprint("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )
    optimizer = get_input_optimizer(input_img)

    eprint("Optimizing..")
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                eprint("run {}:".format(run))
                eprint(
                    "Style Loss : {:4f} Content Loss: {:4f}".format(
                        style_score.item(), content_score.item()
                    )
                )
                eprint()
                if progress_dir:
                    filename = Path(
                        progress_dir, 'progress-{}.png'.format(run[0]))
                    save_image(input_img, filename)

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img
