from fastai.callback.hook import hook_outputs
from torch.nn import Module, MaxPool2d
from torch.nn.functional import l1_loss
from torchvision.models import vgg16_bn


def gram(x):
    bs, c, h, w = x.size()
    x = x.view(bs, c, -1)
    x = x @ x.transpose(1, 2)
    x = x / (c * h * w)
    return x


class ContentLoss(Module):
    def __init__(self, weights, base_loss=l1_loss):
        super().__init__()

        self.weights = weights
        self.base_loss = base_loss

    def forward(self, preds, targs, preds_feats, targs_feats):
        loss = []
        loss += [self.base_loss(preds, targs)]
        loss += [w * self.base_loss(preds_feat, targs_feat)
                 for preds_feat, targs_feat, w in zip(preds_feats, targs_feats, self.weights)]
        return loss


class StyleLoss(Module):
    def __init__(self, weights, base_loss=l1_loss):
        super().__init__()

        self.weights = weights
        self.base_loss = base_loss

    def forward(self, preds_feats, targs_feats):
        loss = []
        loss += [(w ** 2) * 5e3 * self.base_loss(gram(preds_feat), gram(targs_feat))
                 for preds_feat, targs_feat, w in zip(preds_feats, targs_feats, self.weights)]
        return loss


class PerceptualLoss(Module):
    def __init__(self, model_feat, layer_ind, weights,
                 base_loss=l1_loss):
        super().__init__()

        self.model_feat = model_feat
        loss_feats = [self.model_feat[i] for i in layer_ind]
        self.hooks = hook_outputs(loss_feats, detach=False)

        self.content_loss = ContentLoss(weights, base_loss)
        self.style_loss = StyleLoss(weights, base_loss)

    def get_feats(self, x, clone=False):
        self.model_feat(x)
        feats = [(o.clone() if clone else o) for o in self.hooks.stored]
        return feats

    def forward(self, preds, targs):
        preds_feats = self.get_feats(preds)
        targs_feats = self.get_feats(targs, clone=True)

        loss = []
        loss += self.content_loss(preds, targs, preds_feats,
                                  targs_feats)
        loss += self.style_loss(preds_feats, targs_feats)
        loss = sum(loss)
        return loss


def disable_requires_grad(model):
    for p in model.parameters():
        p.requires_grad_(False)


def get_blocks(model):
    blocks = [i-1 for i, o in enumerate(model.children()) if isinstance(o, MaxPool2d)]
    return blocks


class PerceptualLossVGG16(PerceptualLoss):
    def __init__(self):
        model_feat = vgg16_bn(pretrained=True, progress=False).features.cuda().eval()
        disable_requires_grad(model_feat)
        layer_ind = get_blocks(model_feat)[-3:]
        weights = [5, 15, 2]

        super().__init__(model_feat, layer_ind, weights)
