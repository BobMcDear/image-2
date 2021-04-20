from io import BytesIO
from random import choice, randint, random

from PIL import Image, ImageEnhance, ImageFilter
from fastai.data.transforms import get_image_files
from fastai.vision.core import PILImage

from utils import parse_arguments


class Degrader:
    def __init__(self):
        self.tfms = []
        attrs = dir(self)
        for attr in attrs:
            if attr.endswith('_degrader'):
                attr = getattr(self, attr)
                self.tfms.append(attr)

    def __call__(self, img):
        for tfm in self.tfms:
            img = tfm(img)
        return img

    def size_degrader(self, img, p=0.95):
        do = (random() < p)
        if not do:
            return img

        w, h = img.size
        orig_sz = (w, h)
        small_w = randint(w // 2.6, w // 1.4)
        small_h = randint(h // 2.6, h // 1.4)
        small_sz = (small_w, small_h)

        ms = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC]
        m1 = choice(ms)
        m2 = choice(ms)

        img = img.resize(small_sz, m1)
        img = img.resize(orig_sz, m2)
        return img

    def artifact_degrader(self, img, p=0.95):
        do = (random() < p)
        if not do:
            return img

        qs = list(range(15, 20)) + list(range(55, 75))
        q = choice(qs)

        with BytesIO() as f:
            img.save(f, format='JPEG', quality=q)
            f.seek(0)
            img = Image.open(f)
            img.load()
        return img

    def blur_degrader(self, img, p=0.005):
        do = (random() < p)
        if not do:
            return img

        ms = [ImageFilter.GaussianBlur(1),
              ImageFilter.BoxBlur(1)]
        m = choice(ms)

        img = img.filter(m)
        return img

    def color_degrader(self, img, p=0.001):
        do = (random() < p)
        if not do:
            return img

        f = randint(95, 105)/100
        enhancer = ImageEnhance.Color(img)

        img = enhancer.enhance(f)
        return img

    def contrast_degrader(self, img, p=0.001):
        do = (random() < p)
        if not do:
            return img

        f = randint(95, 105) / 100
        enhancer = ImageEnhance.Contrast(img)

        img = enhancer.enhance(f)
        return img

    def brightness_degrader(self, img, p=0.001):
        do = (random() < p)
        if not do:
            return img

        f = randint(95, 105) / 100
        enhancer = ImageEnhance.Brightness(img)

        img = enhancer.enhance(f)
        return img


def main(path='resized/', degraded_path='degraded/'):
    degrader = Degrader()
    fnames = get_image_files(path)
    for fname in fnames:
        img = PILImage.create(fname)
        img = degrader(img)

        degraded_fname = degraded_path + fname.name
        img.save(degraded_fname, quality=95)


if __name__ == '__main__':
    names = ['path', 'degraded_path']
    defaults = ['resized/', 'degraded/']
    args = parse_arguments(names, defaults)

    main(*args)
