from fastai.data.block import DataBlock
from fastai.data.transforms import ColReader, ColSplitter, Normalize
from fastai.vision.augment import aug_transforms
from fastai.vision.data import ImageBlock
from pandas import read_csv


def get_dblock(path='resized/', degraded_path='degraded/'):
    blocks = (ImageBlock, ImageBlock)
    get_x = ColReader('fnames', pref=degraded_path)
    get_y = ColReader('fnames', pref=path)
    splitter = ColSplitter('is_val')
    batch_tfms = [Normalize.from_stats([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225]),
                  *aug_transforms(max_rotate=5, max_warp=0.1,
                                  p_lighting=0.)]

    dblock = DataBlock(blocks=blocks,
                       get_x=get_x,
                       get_y=get_y,
                       splitter=splitter,
                       batch_tfms=batch_tfms)
    return dblock


def get_dls(df='fnames.csv', path='resized/', degraded_path='degraded/',
            bs=2, num_workers=8, n_gpus=1):
    bs = bs // n_gpus
    df = read_csv(df)
    dblock = get_dblock(path, degraded_path)
    dls = dblock.dataloaders(df, bs=bs, num_workers=num_workers)
    dls.c = 3
    return dls
