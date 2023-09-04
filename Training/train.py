import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_custom import get_dataloaders
from utils.checkpoint import save
from utils.loops import train, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(net, logger, hparams):
    # Create dataloaders
    trainloader, valloader, testloader = get_dataloaders(
        path=hparams["database_path"],
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
    )

    net = net.to(device)

    learning_rate = float(hparams["lr"])
    scaler = GradScaler()

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=0.0001,
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.75, patience=5, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("Training", hparams["name"], "on", device)
    for epoch in range(hparams["start_epoch"], hparams["n_epochs"]):
        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(net, valloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        # Update learning rate
        scheduler.step(loss_v)

        if acc_v > best_acc:
            best_acc = acc_v
            save(net, logger, hparams, epoch + 1, optimizer)
            logger.plot(hparams, save=True, show=False)

        if (epoch + 1) % hparams["save_freq"] == 0 or (epoch + 1) == hparams[
            "n_epochs"
        ]:
            save(net, logger, hparams, epoch + 1, optimizer)
            logger.plot(hparams, save=True, show=False)

        print(
            f"Epoch {epoch + 1:02} Train Accuracy: {acc_tr:2.4}, Val Accuracy: {acc_v:2.6}"
        )

    # Calculate performance on test set
    acc_test, loss_test = evaluate(net, testloader, criterion)
    print(f"Test Accuracy: {acc_test:2.4}, Test Loss: {loss_test:2.6}")
