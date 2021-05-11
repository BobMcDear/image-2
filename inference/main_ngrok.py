from gc import collect

from flask import Flask, render_template
from flask_ngrok import run_with_ngrok
from torch import cuda

from get_input import get_input
from model import load_model
from upgrade import compare
from utils import img_to_b64


app = Flask(__name__)
run_with_ngrok(app)
device = 'cuda' if cuda.is_available() else 'cpu'
model = load_model().to(device)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload():
    img, enhancement_level, enlarge = get_input()

    res, img = compare(model, img, enhancement_level,
                       enlarge, device)
    collect()

    res = img_to_b64(res)
    img = img_to_b64(img)
    out = {'res': res, 'img': img}
    return out


if __name__ == "__main__":
    app.secret_key = 'nobody can guess this secret key'
    app.debug = False
    app.use_reloader = False
    app.run()
