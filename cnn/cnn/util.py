import glob
import random

import torchvision.transforms as transforms
import cv2 as cv


def load_ds(base_path):
    bikes = [(f, 0) for f in glob.glob(f'{base_path}/Bike/*')]
    cars = [(f, 1) for f in glob.glob(f'{base_path}/Car/*')]

    return cars + bikes


def load_training_ds(signaller, basepath):

    imgs = load_ds(basepath)
    sigs = []
    for img, label in imgs:
        img = signaller(cv.imread(img))

    random.shuffle(sigs)

    signals = [s for s, l in sigs]
    labels = [l for s, l in sigs]

    return signals, labels


def resnet_transform():
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])


def create_hog_descriptor() -> cv.HOGDescriptor:
    win_size = (96, 96)
    block_size = (32, 32)
    block_stride = (16, 16)
    cell_size = (8, 8)
    nbins = 9
    deriv_aperture = 1
    win_sigma = -1
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = 1
    nlevels = 64
    signed_gradients = True

    return cv.HOGDescriptor(
            win_size,
            block_size,
            block_stride,
            cell_size,
            nbins,
            deriv_aperture,
            win_sigma,
            histogram_norm_type,
            l2_hys_threshold,
            gamma_correction,
            nlevels,
            signed_gradients)
