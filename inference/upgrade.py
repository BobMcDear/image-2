from PIL import Image

from model import load_model


def predict(model, img, scale_factor=2,
            enhancement_level=2, enhance_first=False,
            device='cpu'):
    f = scale_factor ** (1 / enhancement_level)
    res = img

    if enhance_first:
        res = model.predict(res, device)

    for _ in range(enhancement_level):
        w, h = res.size
        new_sz = (int(f * w), int(f * h))
        res = res.resize(new_sz, Image.LANCZOS)
        res = model.predict(res, device)

    return res


def find_scale_factor(img, enlarge=True):
    w, h = img.size
    mn, mx = min(w, h), max(w, h)

    if (not enlarge) or (512 <= mx):
        scale_factor = 1

    elif mn < 100:
        scale_factor = 3.5
    
    elif mn < 160:
        scale_factor = 3
    
    elif mn < 200:
        scale_factor = 2.5

    elif mn < 256:
        scale_factor = 2

    else:
        scale_factor = min(512/w, 512/h)

    return scale_factor


def upgrade(model, img, enhancement_level=2, 
            enlarge=True, device='cpu'):
    scale_factor = find_scale_factor(img, enlarge)
    enhance_first = ((3 <= enhancement_level) and (3.4 < scale_factor))

    res = predict(model, img, scale_factor,
                  enhancement_level, enhance_first,
                  device)

    return res


def compare(img, enhancement_level=3, enlarge=True, device='cpu'):
    model = load_model('models/model.pt').to(device)
    res = upgrade(model, img, enhancement_level=enhancement_level,
                  enlarge=enlarge, device=device)
    new_sz = res.size
    img = img.resize(new_sz, Image.LANCZOS)
    return res, img
