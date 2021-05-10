from flask import request

from utils import b64_to_img


def get_input():
    f = request.form['file']
    img = b64_to_img(f)
    enhancement_level = int(request.form['enhancement_level'])
    enlarge = (request.form['enlarge'] == 'true')
    return img, enhancement_level, enlarge
