from data import get_dls
from model import get_unet
from train import train
from utils import parse_arguments


def main(df='fnames.csv', path='resized/', degraded_path='degraded/', bs=2, num_workers=8,
         model_name='swsl_resnet18', n_gpus=1,
         frozen_epochs=5, unfrozen_epochs=5):
    print('Getting the data...')
    dls = get_dls(df, path, degraded_path, bs,
                  num_workers, n_gpus)

    print('Getting the learner...')
    learn = get_unet(dls, model_name)
    train(learn, bs, frozen_epochs,
          unfrozen_epochs, n_gpus)


if __name__ == '__main__':
    names = ['df', 'path', 'degraded_path', 'bs', 'num_workers', 'model_name',
             'n_gpus', 'frozen_epochs', 'unfrozen_epochs']
    defaults = ['fnames.csv', 'resized/', 'degraded/', 2, 8, 'swsl_resnet18', 1, 5, 5]
    args = parse_arguments(names, defaults)

    main(*args)
