import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2 as cv

from networks import SmallCNN, LargeCNN
import util


class HOGClassifier:
    def __init__(self, model):
        self.model = None
        self.hog = util.create_hog_descriptor()

        if isinstance(model, str):
            self.model = cv.ml.SVM.load(model)
        elif isinstance(model, cv.ml.SVM):
            self.model = model
        else:
            raise TypeError('Invalid value for argument model',
                            f'({type(model)})')

    def predict(self, img):
        hog = self.hog.compute(img)
        pred = self.model.predict(np.matrix(hog))[1][0][0]  # > 0.5
        pred = int(pred)
        # print(pred)
        return pred

    def __call__(self, img):
        return self.predict(img)


class CNNClassifier:
    def predict(self, img):
        img = Image.fromarray(np.uint8(img))
        transformed = self.transform(img).unsqueeze(0)
        prob, label = F.softmax(self.model(transformed), dim=1).topk(1)
        label = int(label[0][0])

        return label

    def __call__(self, img):
        return self.predict(img)


class CustomCNNClassifier(CNNClassifier):
    def __init__(self, model, model_cls):
        super().__init__()
        self.model = None

        if isinstance(model, str):
            self.model = model_cls.from_file(model)
        elif isinstance(model, model_cls):
            self.model = model
        else:
            raise TypeError('Invalid value for argument model',
                            f'({type(model)})')

        self.model.eval()
        self.transform = util.cnn_transform()


class SmallCNNClassifier(CustomCNNClassifier):
    def __init__(self, model):
        super().__init__(model, SmallCNN)


class LargeCNNClassifier(CustomCNNClassifier):
    def __init__(self, model):
        super().__init__(model, LargeCNN)


class ResnetClassifier(CNNClassifier):
    def __init__(self, model):
        super().__init__()
        self.model = None

        if isinstance(model, str):
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 2)
            self.model.load_state_dict(torch.load(model))
        elif isinstance(model, models.ResNet):
            self.model = model
        else:
            raise TypeError('Invalid value for argument model')

        self.model.eval()
        self.transform = util.resnet_transform()
