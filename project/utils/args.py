from pathlib import Path
from project.utils.cli import eprint


class InvalidCopyStyleArgs(Exception):
    pass


class CopyStyleArgs:
    def __init__(
        self,
        image: str,
        style_image: str,
        output: str = "",
        resize: bool = False,
        progress_dir: str = "",
        max_size: bool = False,
    ):
        image = Path(image)
        style_image = Path(style_image)
        if progress_dir:
            progress_dir = Path(progress_dir)
        validity, err = self._valid_args(
            image, style_image, output, resize, progress_dir, max_size
        )

        if not validity:
            raise InvalidCopyStyleArgs(err)

        self.resize = resize
        self.image_path = image
        self.style_image = style_image
        self.progress_dir = progress_dir
        self.max_size = max_size

        if output:
            _opath = Path(output)
            if not _opath.suffix:
                self.output_name = output + ".png"
            else:
                self.output_name = output
        else:
            self.output_name = (
                image.stem + "-style-copy-" + style_image.stem + image.suffix
            )

    def _valid_args(
        self, image: Path, copy_style_image: Path, _output, _resize, progress_dir: Path, _max_size
    ) -> (bool, str):
        if not image.exists():
            raise InvalidCopyStyleArgs(
                "Given image path does not exist:\n\t{}".format(str(image))
            )

        if not copy_style_image.exists():
            raise InvalidCopyStyleArgs(
                "Given image path does not exist:\n\t{}".format(
                    str(copy_style_image))
            )

        if not progress_dir.exists():
            eprint("Created new directory:\n\t" + str(progress_dir))
            progress_dir.mkdir(parents=True, exist_ok=True)

        return (True, "")

    def __repr__(self):
        return 'CopyStyleArgs("{0}", "{1}", output="{2}")'.format(
            self.image_path, self.style_image, self.output_name
        )

    def __str__(self):
        return "python main.py {0} --style {1} --output {2} --resize={3} --progress={4} --max-size={5}".format(
            self.image_path, self.style_image, self.output_name,
            self.resize, self.progress_dir, self.max_size
        )
