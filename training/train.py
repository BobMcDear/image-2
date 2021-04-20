from contextlib import nullcontext

from fastai.callback.schedule import fit_one_cycle
from fastai.distributed import distrib_ctx


def modify_lr(bs=2):
    lr_div = 32 / bs
    frozen_lr = 4e-4 / lr_div
    unfrozen_lr = slice(1e-5 / lr_div, 1e-4 / lr_div)
    lrs = (frozen_lr, unfrozen_lr)
    return lrs


def fit(learn, frozen_lr, unfrozen_lr,
        frozen_epochs=5, unfrozen_epochs=5):
    learn.fit_one_cycle(frozen_epochs, frozen_lr)
    learn.save('frozen')

    learn.unfreeze()
    learn.fit_one_cycle(unfrozen_epochs, unfrozen_lr)
    learn.save('unfrozen')
    learn.export('learn.pkl')


def train(learn, bs=2, frozen_epochs=5,
          unfrozen_epochs=5, n_gpus=1):
    frozen_lr, unfrozen_lr = modify_lr(bs)
    ctx = learn.distrib_ctx if (1 < n_gpus) else nullcontext
    with ctx():
        fit(learn, frozen_lr, unfrozen_lr,
            frozen_epochs, unfrozen_epochs)
