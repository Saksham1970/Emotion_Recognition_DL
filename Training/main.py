from utils.hparams import setup_hparams
from utils.setup_net import setup_net
from train import run


hparams = {
    "net": "VGG_16",
    "name": "VGG_16_CustomFaceData_Color_SGD_RLRP_min",
    "batch_size": 32,
    "n_epochs": 150,
    "database_path": "./Datasets/CustomFaceData",
    "num_classes": 8,
    "restore_epoch": None,
}

hparams = setup_hparams(**hparams)
logger, net = setup_net(hparams)


run(net, logger, hparams)
