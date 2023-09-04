import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

current_model = 1
net = None
color = None

names = ["Custom Face Data", "FER 2013", "FER plus"]
colors = [True, False, False]

label_names = np.array(
    [
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happiness",
        "neutral",
        "sadness",
        "surprise",
    ]
)


def load_model(path):
    global net
    net = torch.jit.load(path)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()
    net = net.to(device)


def next_model(model=None):
    global current_model, color
    if model:
        current_model = model
    else:
        current_model = (current_model + 1) % len(names)
    load_model("./Trained Models/" + names[current_model] + ".pt")
    color = colors[current_model]


def get_transform():
    mu, st = 0, 255

    if not color:
        return transforms.Compose(
            [
                transforms.Lambda(
                    lambda face: Image.fromarray(face).convert("L").resize((48, 48))
                ),
                transforms.TenCrop(40),
                transforms.Lambda(
                    lambda crops: torch.stack(
                        [transforms.ToTensor()(crop) for crop in crops]
                    )
                ),
                transforms.Lambda(
                    lambda tensors: torch.stack(
                        [
                            transforms.Normalize(mean=(mu,), std=(st,))(t)
                            for t in tensors
                        ]
                    )
                ),
            ]
        )

    else:
        return transforms.Compose(
            [
                transforms.Lambda(lambda face: cv2.cvtColor(face, cv2.COLOR_BGR2RGB)),
                transforms.Lambda(
                    lambda face: Image.fromarray(face).convert("RGB").resize((96, 96))
                ),
                transforms.TenCrop(80),
                transforms.Lambda(
                    lambda crops: torch.stack(
                        [transforms.ToTensor()(crop) for crop in crops]
                    )
                ),
                transforms.Lambda(
                    lambda tensors: torch.stack(
                        [
                            transforms.Normalize(mean=(mu,), std=(st,))(t)
                            for t in tensors
                        ]
                    )
                ),
            ]
        )


def predict(inputs, top_n=1, NCrops=True):
    inputs = inputs.to(device)
    if NCrops:
        # fuse crops and batchsize
        original_shape = inputs.shape
        inputs = inputs.view(-1, *inputs.shape[-3:])

        # forward
        outputs = net(inputs)

        # combine results across the crops
        outputs = outputs.view(*original_shape[:2], -1)
        outputs = torch.sum(outputs, dim=1) / original_shape[1]
    else:
        outputs = net(inputs)

    _, preds = torch.topk(outputs.data, top_n, dim=1)
    return label_names[np.array(preds.cpu())]


def working_model():
    return names[current_model]


next_model()
