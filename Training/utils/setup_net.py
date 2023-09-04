from vgg import VGG
from utils.checkpoint import restore
from utils.logger import Logger

archs = {
    "VGG_11": ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    "VGG_16": ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
}


def setup_net(hparams):
    net = VGG(arch=archs[hparams["net"]], lr=hparams["lr"], drop=hparams["drop"])

    # Prepare logger
    logger = Logger()
    if hparams["restore_epoch"]:
        restore(net, logger, hparams)

    return logger, net
