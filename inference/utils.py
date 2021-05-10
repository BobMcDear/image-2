from base64 import b64decode, b64encode
from io import BytesIO

from numpy import array, moveaxis, uint8
from PIL import Image


def torch_to_np(a):
    a = a.to('cpu')
    a = array(a)
    a = moveaxis(a, 0, -1)
    a = uint8(a)
    return a


def torch_to_img(a):
    a = torch_to_np(a)
    a = Image.fromarray(a)
    return a


def b64_to_img(b64):
    b64 = BytesIO(b64decode(b64))
    img = Image.open(b64).convert('RGB')
    return img


def img_to_b64(img):
    data = BytesIO()
    img.save(data, format='JPEG', quality=95)
    data = data.getvalue()
    img = b64encode(data).decode('utf-8')
    return img