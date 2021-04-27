from flask import request
from PIL import Image


def get_input():
    f = request.files['file']
    img = Image.open(f).convert('RGB')

    enhancement_level = int(request.form['enhancement_level'])
    enlarge = request.form.get('enlarge')
    inp = (img, enhancement_level, enlarge)
    return inp
