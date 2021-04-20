from fastai.data.transforms import get_image_files
from fastai.vision.augment import Resize
from PIL import Image

from utils import parse_arguments


def main(path='images/', resized_path='resized/',
         size=256):
    resize = Resize(size, resamples=(Image.LANCZOS,
                                     Image.LANCZOS))

    fnames = get_image_files(path)
    for fname in fnames:
        img = Image.open(fname).convert('RGB')
        img = resize(img)

        resized_fname = resized_path + fname.name
        img.save(resized_fname, quality=95)


if __name__ == '__main__':
    names = ['path', 'resized_path', 'size']
    defaults = ['images/', 'resized/', 256]
    args = parse_arguments(names, defaults)

    main(*args)
