import os
import torch


def save(net, logger, hparams, epoch, optimizer=None):
    # Create the path the checkpint will be saved at using the epoch number
    path = os.path.join(hparams["model_save_dir"], "epoch_" + str(epoch))

    # create a dictionary containing the logger info and model info that will be saved
    checkpoint = {
        "logs": logger.get_logs(),
        "params": net.state_dict(),
    }

    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # save checkpoint
    torch.save(checkpoint, path)


def restore(
    net,
    logger,
    hparams,
    optimizer=None,
):
    """Load back the model and logger from a given checkpoint
    epoch detailed in hps['restore_epoch'], if available"""
    path = os.path.join(
        hparams["model_save_dir"], "epoch_" + str(hparams["restore_epoch"])
    )

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)

            logger.restore_logs(checkpoint["logs"])
            net.load_state_dict(checkpoint["params"])

            if "optimizer_state_dict" in checkpoint and optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Using the Optimizer params that were found.")

            print("Net Restored!")

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e)
            hparams["start_epoch"] = 0

    else:
        print("Restore point unavailable. Training from scratch.")
        hparams["start_epoch"] = 0
