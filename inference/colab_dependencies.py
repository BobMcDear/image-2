from os import chdir, system

from gdown import download


def install_libraries():
    libs = ['pip install -U --no-deps fastai==2.3.1',
            'pip install -U --no-deps fastcore==1.3.20',
            'pip install flask-ngrok==0.0.25',
            'pip install timm==0.4.5']
    for lib in libs:
        system(lib)


def download_model():
    url = 'https://drive.google.com/u/0/uc?id=1SVxl-UjFZXDoZu2h0yZadkErOruEfiwl'
    output = 'models/model.pt'
    download(url, output, quiet=True)


def main():
    install_libraries()
    system('mkdir models/')
    download_model()


if __name__ == '__main__':
    main()
