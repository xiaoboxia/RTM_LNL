

from torch import nn

from option import args
def network_choose():
    if args.model == 'FC':
        from networkbk import fc_network
        return fc_network.FCNetowrk().to(args.device)
