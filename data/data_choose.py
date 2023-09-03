from option import args


def get_data(flip_percentage):
    if args.dataset == 'mnist':
        from data.mnist_data import get_mnist_data
        return get_mnist_data(flip_percentage)
