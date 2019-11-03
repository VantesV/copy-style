import sys
import traceback
import time
import argparse

from project.utils.args import CopyStyleArgs
from project.lib import run
from project.utils.cli import eprint


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Copy art styles to images")
    parser.add_argument("IMAGE", help="the image you want transformed")
    parser.add_argument(
        "-s", "--style", help="the piece where the style is copied from", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="the output file. If this is not provided then [IMAGE].style-copied is used",
        required=True,
    )
    parser.add_argument(
        "-r", "--resize", help="resize the image to the copy image", action="store_true"
    )
    parser.add_argument("-p", "--progress",
                        help="directory to output progress")
    parser.add_argument(
        "-m", "--max-size", help="use IMSIZE 512 regarldess of GPU availibilyt", action="store_true")

    args = parser.parse_args()
    image = args.IMAGE
    resize = args.resize
    style_image = args.style
    progress_dir = args.progress
    max_size = args.max_size
    error = 0
    if not image:
        eprint("Image cannot be empty!")
        sys.exit(1)
    if not style_image:
        eprint("Copy cannot be empty!")
        sys.exit(1)

    try:
        args = CopyStyleArgs(
            image,
            style_image,
            output=args.output,
            resize=resize,
            progress_dir=progress_dir,
            max_size=max_size
        )
        run(args)
    except Exception as e:
        error = 1
        eprint(str(e))
        traceback.print_exc()

    if error == 1:
        sys.exit(error)
    end = time.time()
    print("Time elapsed: " + str(end - start) + "seconds")

if __name__ == "__main__":
    main()
