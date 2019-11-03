# Copy Style

## Setup

Install [pipenv](https://pipenv.readthedocs.io/en/latest/).

Run `pipenv install`

Download a pretrained model:

```bash
$ wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
```

### How to run

```bash
$ pipenv shell
$ python main.py \
some-input-img.jpg \
--style some-style-img.jgp \
--o name-of-output-file.png \
--resize \
--progress path-to-progress-dir
```

```
usage: main.py [-h] -s STYLE -o OUTPUT [-r] [-p PROGRESS] IMAGE

Copy art styles to images

positional arguments:
  IMAGE                 the image you want transformed

optional arguments:
  -h, --help            show this help message and exit
  -s STYLE, --style STYLE
                        the piece where the style is copied from
  -o OUTPUT, --output OUTPUT
                        the output file. If this is not provided then
                        [IMAGE].style-copied is used
  -r, --resize          resize the image to the copy image
  -p PROGRESS, --progress PROGRESS
                        directory to output progress

```

## About

Refer to https://arxiv.org/abs/1508.06576