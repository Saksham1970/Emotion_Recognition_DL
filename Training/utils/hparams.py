import os

hparams_template = {
    "net": "",  # which net do you want to train (VGG_11/VGG_16)
    "name": "",  # whatever you want to name your run
    "n_epochs": 300,
    "model_save_dir": None,  # where will checkpoints be stored
    "restore_epoch": None,  # continue training from a specific saved point
    "start_epoch": 0,
    "lr": 0.01,  # starting learning rate
    "save_freq": 20,  # how often to create checkpoints
    "drop": 0.2,
    "batch_size": 64,
    "database_path": "./Datasets/FERplus",
    "num_workers": 8,
    "num_classes": 8,
}

possible_nets = ["VGG_11", "VGG_16"]


def setup_hparams(**kwargs):
    hparams = hparams_template.copy()

    for key, value in kwargs.items():
        if key not in hparams:
            raise ValueError(key + " is not a valid hyper parameter")
        else:
            hparams[key] = value

    # Invalid net check
    if hparams["net"] not in possible_nets:
        raise ValueError(
            "Invalid net.\nPossible ones include:\n - " + "\n - ".join(possible_nets)
        )

    # invalid parameter check
    try:
        hparams["n_epochs"] = int(hparams["n_epochs"])
        hparams["start_epoch"] = int(hparams["start_epoch"])
        hparams["save_freq"] = int(hparams["save_freq"])
        hparams["lr"] = float(hparams["lr"])
        hparams["drop"] = float(hparams["drop"])
        hparams["batch_size"] = int(hparams["batch_size"])

        if hparams["restore_epoch"]:
            hparams["restore_epoch"] = int(hparams["restore_epoch"])
            hparams["start_epoch"] = int(hparams["restore_epoch"])

        # make sure we can checkpoint regularly or at least once (at the end)
        if hparams["n_epochs"] < 20:
            hparams["save_freq"] = min(5, hparams["n_epochs"])

    except Exception as e:
        raise ValueError("Invalid input parameters")

    # create checkpoint directory

    if not hparams["model_save_dir"]:
        hparams["model_save_dir"] = os.path.join(
            os.getcwd(), "checkpoints", hparams["name"]
        )

        if not os.path.exists(hparams["model_save_dir"]):
            os.makedirs(hparams["model_save_dir"])

    return hparams
