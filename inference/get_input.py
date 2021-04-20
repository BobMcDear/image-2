from flask import request
from PIL import Image


def verify_extension(f):
    fname = f.filename
    extension = fname.split('.')[-1]
    allowed = ['jpg', 'jpeg', 'png']
    if extension in allowed:
        return True
    return False


def verify_size(img):
    w, h = img.size
    mx = max(w, h)
    if 768 <= mx:
        return False
    return True


def get_input():
    f = request.files['file']
    if not verify_extension(f):
        return 'extension', None, None

    img = Image.open(f).convert('RGB')
    if not verify_size(img):
        return 'size', None, None

    enhancement_level = int(request.form['enhancement_level'])
    enlarge = request.form.get('enlarge')
    inp = (img, enhancement_level, enlarge)
    return inp
