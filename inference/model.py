from functools import partial

from fastai.data.transforms import IntToFloatTensor, Normalize
from fastai.layers import NormType
from fastai.torch_core import TensorImage
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.learner import create_body
from fastcore.transform import Pipeline
from timm import create_model
from torch import load as torch_load, no_grad

from utils import torch_to_img


def get_model():
    encoder = create_body(partial(create_model, 'ecaresnet101d_pruned'),
                          pretrained=False)
    n_out = 3
    img_size = (128, 128)
    blur = True
    self_attention = True
    norm_type = NormType.Weight
    model = DynamicUnet(encoder, n_out=n_out, img_size=img_size,
                        blur=blur, self_attention=self_attention,
                        norm_type=norm_type)
    return model


def predict(self, img, device='cuda'):
    a = TensorImage(img).to(device).movedim(-1, 0)
    a = self.tfms(a)

    with no_grad():
        a = self(a)

    a = self.tfms.decode(a)
    a.squeeze_(0)

    img = torch_to_img(a)
    return img


def load_model(path='models/model.pt'):
    DynamicUnet.predict = predict
    model = get_model()
    model.load_state_dict(torch_load(path))
    model.eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    funcs = [IntToFloatTensor(),
             Normalize.from_stats(mean, std)]
    tfms = Pipeline(funcs=funcs)
    model.tfms = tfms
    return model
