from functools import partial

from fastai.callback.fp16 import to_fp16
from fastai.layers import NormType
from fastai.optimizer import RMSProp
from fastai.torch_core import params
from fastai.vision.learner import model_meta, unet_learner
from fastcore.foundation import L
from timm import create_model

from loss import PerceptualLossVGG16


def resnet_split(m):
    m1 = m[0][:6]
    m2 = m[0][6:]
    m3 = m[1:]
    split = L(m1, m2, m3).map(params)
    return split


def get_arch(model_name='swsl_resnet18'):
    arch = partial(partial(create_model),
                   model_name)
    cut = None
    stats = None
    resnet_meta = {'cut': cut,
                   'split': resnet_split,
                   'stats': stats}
    model_meta[arch] = resnet_meta
    return arch


def get_unet(dls, model_name='swsl_resnet18'):
    arch = get_arch(model_name)
    loss_func = PerceptualLossVGG16()
    opt_func = partial(RMSProp, mom=0.9, wd=1e-1)
    blur = True
    self_attention = True
    norm_type = NormType.Weight
    learn = unet_learner(dls, arch, normalize=False, loss_func=loss_func,
                         opt_func=opt_func, blur=blur,
                         self_attention=self_attention,
                         norm_type=norm_type).to_fp16()
    return learn
