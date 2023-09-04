import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample


def prepare_data(dataset_path, num_imgs=None):
    """Prepare data for modeling
    output: image and label array"""

    label_names = {
        "anger": 0,
        "contempt": 1,
        "disgust": 2,
        "fear": 3,
        "happiness": 4,
        "neutral": 5,
        "sadness": 6,
        "surprise": 7,
    }

    types = ["train", "test"]
    output = []

    for type_ in types:
        labels = []
        images = []

        path_ = os.path.join(dataset_path, type_)
        for emotion in os.listdir(path_):
            emotion_path = os.path.join(path_, emotion)
            for image in os.listdir(emotion_path)[:num_imgs]:
                labels.append(label_names[emotion])
                images.append(
                    Image.open(os.path.join(emotion_path, image)).convert("L")
                )

        np_images = np.empty(len(images), dtype="object")
        np_images[:] = images

        output.append([np_images, np.array(labels)])

    test = output.pop(-1)
    X_val, X_test, y_val, y_test = train_test_split(
        *test, test_size=0.5, random_state=42, stratify=test[-1]
    )
    output.append([X_val, y_val])
    output.append([X_test, y_test])

    return output


def get_dataloaders(path, batch_size=64, augment=True, num_workers=0):
    """Prepare Train & Test dataloaders
    Augment training data using:
        - cropping
        - shifting (vertical/horizental)
        - horizental flipping
        - rotation

    input: path to FER+ Folder
    output: (Dataloader, Dataloader Dataloader)"""

    (xtrain, ytrain), (xval, yval), (xtest, ytest) = prepare_data(path)

    mu, st = 0, 255

    test_transform = transforms.Compose(
        [
            transforms.TenCrop(40),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(
                lambda tensors: torch.stack(
                    [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors]
                )
            ),
        ]
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
                transforms.RandomApply(
                    [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
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
                transforms.Lambda(
                    lambda tensors: torch.stack(
                        [transforms.RandomErasing(p=0.5)(t) for t in tensors]
                    )
                ),
            ]
        )
    else:
        train_transform = test_transform

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    valloader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    testloader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return trainloader, valloader, testloader
