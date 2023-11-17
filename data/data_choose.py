from option import args

def get_data(flip_percentage):
    if args.dataset == 'mnist':
        from data.mnist_data import get_mnist_data
        return get_mnist_data(flip_percentage)
    elif args.dataset == 'cifar10':
        from data.cifar10_data import get_cifar10_data
        return get_cifar10_data(flip_percentage)
    elif args.dataset == 'svhn':
        from data.svhn_data import get_svhn_data
        return get_svhn_data(flip_percentage)
    elif args.dataset == 'news':
        from data.news_data import get_news_data
        return get_news_data(flip_percentage)    
    elif args.dataset == 'cifar100':
        from data.cifar100_data import get_cifar100_data
        return get_cifar100_data(flip_percentage)
    else:
        raise NotImplementedError