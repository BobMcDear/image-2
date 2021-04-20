from argparse import ArgumentParser


def parse_arguments(names, defaults):
    parser = ArgumentParser()
    for name, default in zip(names, defaults):
        parser.add_argument('--' + name, dest=name,
                            action='store', default=default)

    args = parser.parse_args()
    _args = []
    for name in names:
        arg = getattr(args, name)
        if isinstance(arg, str) and arg.isdigit():
            arg = int(arg)
        _args.append(arg)
    args = _args
    return args
