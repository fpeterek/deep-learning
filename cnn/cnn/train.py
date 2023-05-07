import click
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import cv2 as cv
import numpy as np

import util
from torch_ds import CarBikeDS
from networks import SmallCNN, LargeCNN


@click.command()
@click.option('--model-name', help='Name of model')
def train_resnet(model_name):
    resnet18 = models.resnet18(pretrained=True)

    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)

    transform = util.resnet_transform()

    batch_size = 32

    trainset = CarBikeDS(car_dir='data/resnet/train/cars/',
                         bike_dir='data/resnet/train/bikes/',
                         transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    best_loss = 1.0
    best_model = None

    for epoch in range(8):

        print(f'{epoch=}')

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print(f'[{epoch + 1}, {i + 1:5d}]',
                      f'loss: {running_loss / 20:.3f}')
                if running_loss < best_loss:
                    best_model = resnet18.state_dict().copy()
                    best_loss = running_loss
                running_loss = 0.0

    print('Finished Training')

    if best_model:
        torch.save(best_model, model_name)


def train_cnn(network, dataset: str, model_name: str,
              batch_size=64, lr=0.001):

    trainset = CarBikeDS(car_dir=f'{dataset.strip("/")}/train/cars',
                         bike_dir=f'{dataset.strip("/")}/train/bikes',
                         transform=util.cnn_transform())
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)

    best_loss = 10.0
    best_model = None

    for epoch in range(10):  # loop over the dataset multiple times

        print(f'{epoch=}')

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print(f'[{epoch + 1}, {i + 1:5d}]',
                      f'loss: {running_loss / 20:.3f}')
                if running_loss < best_loss or best_model is None:
                    best_model = network.state_dict().copy()
                    best_loss = running_loss
                running_loss = 0.0

    print('Finished Training')

    if best_model:
        torch.save(best_model, model_name)


@click.command()
@click.option('--dataset', help='Dataset')
@click.option('--model-name', help='Output path')
@click.option('--batch-size', help='Batch size', default=64)
@click.option('--lr', help='Learning rate', default=0.001)
def train_small(dataset: str, model_name: str, batch_size, lr):
    train_cnn(SmallCNN(), dataset, model_name, batch_size, lr)


@click.command()
@click.option('--dataset', help='Dataset')
@click.option('--model-name', help='Output path')
@click.option('--batch-size', help='Batch size', default=64)
@click.option('--lr', help='Learning rate', default=0.001)
def train_large(dataset: str, model_name: str, batch_size, lr):
    train_cnn(LargeCNN(), dataset, model_name, batch_size, lr)


@click.command()
@click.option('--model-name', help='Name of the model')
@click.option('--c', default=100, help='C parameter of svm')
@click.option('--gamma', default=1.0, help='Gamma parameter of svm')
def train_hog(model_name: str, c: float = 100.0, gamma: float = 1.0):
    hog = util.create_hog_descriptor()

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_INTER)
    svm.setC(c)
    svm.setGamma(gamma)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    signals, labels = util.load_training_ds(signaller=hog.compute,
                                            basepath='data/96/train/')

    signals = np.matrix(signals)
    labels = np.array(labels)

    svm.train(signals, cv.ml.ROW_SAMPLE, labels)

    svm.save(model_name)


@click.group('Training')
def main() -> None:
    pass


main.add_command(train_hog)
main.add_command(train_resnet)
main.add_command(train_small)
main.add_command(train_large)


if __name__ == '__main__':
    main()
