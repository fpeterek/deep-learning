import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import cv2 as cv
import numpy as np

import util
from torch_ds import CarBikeDS


def train_resnet(free_set, occupied_set, model_name):
    resnet18 = models.resnet18(pretrained=True)

    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_features, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)

    transform = util.resnet_transform()

    batch_size = 4

    trainset = CarBikeDS(car_dir='data/resnet/cars/',
                         bike_dir='data/resnet/bikes/',
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
            if i % 200 == 199:
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                if running_loss < best_loss:
                    best_model = resnet18.state_dict().copy()
                running_loss = 0.0

    print('Finished Training')

    if best_model:
        torch.save(best_model, model_name)


def train_hog(ds: str, model_name: str, c: float = 100.0, gamma: float = 1.0):
    hog = util.create_hog_descriptor()

    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_INTER)
    svm.setC(c)
    svm.setGamma(gamma)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    signals, labels = util.load_training_ds(signaller=hog.compute, basepath=ds)

    signals = np.matrix(signals)
    labels = np.array(labels)

    svm.train(signals, cv.ml.ROW_SAMPLE, labels)

    svm.save(model_name)
