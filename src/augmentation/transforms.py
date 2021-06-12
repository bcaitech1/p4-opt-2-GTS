"""Image transformations for augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import random
from typing import Callable, Dict, Tuple

import PIL
from PIL.Image import Image
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
import torchvision.transforms.functional as F

from src.utils.data import get_rand_bbox_coord

FILLCOLOR = (128, 128, 128)
FILLCOLOR_RGBA = (128, 128, 128, 128)


def transforms_info() -> Dict[
    str, Tuple[Callable[[Image, float], Image], float, float]
]:
    """Return augmentation functions and their ranges."""
    transforms_list = [
        (Identity, 0.0, 0.0),
        (Invert, 0.0, 0.0),
        (Contrast, 0.0, 0.9),
        (AutoContrast, 0.0, 0.0),
        (Rotate, 0.0, 30.0),
        (TranslateX, 0.0, 150 / 331),
        (TranslateY, 0.0, 150 / 331),
        (Sharpness, 0.0, 0.9),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (Color, 0.0, 0.9),
        (Brightness, 0.0, 0.9),
        (Equalize, 0.0, 0.0),
        (Solarize, 256.0, 0.0),
        (Posterize, 8, 4),
        (Cutout, 0, 0.5),
    ]
    return {f.__name__: (f, low, high) for f, low, high in transforms_list}


class Identity:
    """Identity map."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return img


class Invert:
    """Invert the image."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude

    def __call__(self, img: Image):
        return PIL.ImageOps.invert(img)


class Contrast:
    """Put contrast effect on the image."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return PIL.ImageEnhance.Contrast(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


class AutoContrast:
    """Put contrast effect on the image."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
    
    def __call__(self, img:Image):
        return PIL.ImageOps.autocontrast(img)


class Rotate:
    """Rotate the image (degree)."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
    
    def __call__(self, img:Image):
        rot = img.convert("RGBA").rotate(self.magnitude)
        return PIL.Image.composite(
            rot, PIL.Image.new("RGBA", rot.size, FILLCOLOR_RGBA), rot
        ).convert(img.mode)


class TranslateX:
    """Translate the image on x-axis."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return img.transform(
            img.size,
            PIL.Image.AFFINE,
            (1, 0, self.magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=FILLCOLOR,
        )


class TranslateY:
    """Translate the image on y-axis."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return img.transform(
            img.size,
            PIL.Image.AFFINE,
            (1, 0, 0, 0, 1, self.magnitude * img.size[1] * random.choice([-1, 1])),
            fillcolor=FILLCOLOR,
        )


class Sharpness:
    """Adjust the sharpness of the image."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return PIL.ImageEnhance.Sharpness(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


class ShearX:
    """Shear the image on x-axis."""
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return img.transform(
            img.size,
            PIL.Image.AFFINE,
            (1, self.magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            PIL.Image.BICUBIC,
            fillcolor=FILLCOLOR,
        )


class ShearY:
    """Shear the image on y-axis."""
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return img.transform(
            img.size,
            PIL.Image.AFFINE,
            (1, 0, 0, self.magnitude * random.choice([-1, 1]), 1, 0),
            PIL.Image.BICUBIC,
            fillcolor=FILLCOLOR,
        )


class Color:
    """Adjust the color balance of the image."""
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return PIL.ImageEnhance.Color(img).enhance(1 + self.magnitude * random.choice([-1, 1]))


class Brightness:
    """Adjust brightness of the image."""
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return PIL.ImageEnhance.Brightness(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


class Equalize:
    """Equalize the image."""
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def __call__(self, img: Image):   
        return PIL.ImageOps.equalize(img)


class Solarize:
    """Solarize the image."""
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img: Image):
        return PIL.ImageOps.solarize(img, self.magnitude)


class Posterize:
    """Posterize the image."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude

    def __call__(self, img: Image):
        self.magnitude = int(self.magnitude)
        return PIL.ImageOps.posterize(img, self.magnitude)


class Cutout:
    """Cutout some region of the image."""
    def __init__(self, magnitude: float):
        self.magnitude = magnitude

    def __call__(self, img: Image):
        if self.magnitude == 0.0:
            return img
        w, h = img.size
        xy = get_rand_bbox_coord(w, h, self.magnitude)

        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, fill=FILLCOLOR)
        return img


class SquarePad:
    """Square pad to make torch resize to keep aspect ratio."""

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")
