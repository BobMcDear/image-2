from gc import collect

from flask import flash, Flask, redirect, render_template, url_for

from get_input import get_input
from upgrade import compare
from utils import img_to_b64


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload():
    img, enhancement_level, enlarge = get_input()

    if img == 'extention':
        flash('File has to be .jpg, .jpeg, or .png')
        return redirect(url_for('index'))
    
    elif img == 'size':
        flash('Maximum dimension of image must be under 768')
        return redirect(url_for('index'))
    
    res, img = compare(img, enhancement_level, enlarge)
    collect()
    res = img_to_b64(res)
    img = img_to_b64(img)
    return render_template('index.html', img=img, res=res)


if __name__ == "__main__":
    app.secret_key = 'nobody can guess this secret key'
    app.debug = True
    app.run()
