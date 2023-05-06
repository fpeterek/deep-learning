import random
import os

import scipy as sp
import cv2 as cv

import util


def img_96(img):
    img = cv.imread(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.resize(img, (96, 96))


def rotate(img):
    avg = 0
    height, width = img.shape[0], img.shape[1]
    for x in range(width):
        for y in range(height):
            avg += img[y][x]

    avg /= width*height

    angle = random.randint(-40, 40)
    return sp.ndimage.rotate(img, angle, reshape=False, cval=avg)


def data_enhancement(img):
    img = img_96(img)
    rot = rotate(img)
    return [img, rot]


def resnet_img(img):
    img = cv.imread(img)
    return cv.resize(img, (224, 224))


def save_imgs(imgs, parent, imtype):
    parent = parent.strip('/')
    base_folder = f'{parent}/{imtype}'

    os.makedirs(base_folder, exist_ok=True)

    for idx, img in enumerate(imgs):
        path = f'{base_folder}/{idx}.png'
        cv.imwrite(path, img)


def create_96(cars, bikes):
    cars = [img_96(car) for car in cars]
    bikes = [img_96(bike) for bike in bikes]
    save_imgs(cars, 'data/96/', 'cars')
    save_imgs(bikes, 'data/96/', 'bikes')


def create_resnet(cars, bikes):
    cars = [resnet_img(car) for car in cars]
    bikes = [resnet_img(bike) for bike in bikes]
    save_imgs(cars, 'data/resnet/', 'cars')
    save_imgs(bikes, 'data/resnet/', 'bikes')


def create_enhanced(cars, bikes):
    c = []
    b = []

    for car in cars:
        c += data_enhancement(car)

    for bike in bikes:
        b += data_enhancement(bike)

    save_imgs(c, 'data/enhanced/', 'cars')
    save_imgs(b, 'data/enhanced/', 'bikes')


def main():
    data = util.load_ds('data/Car-Bike-Dataset/')

    cars = [f for f, t in data if t == 1]
    bikes = [f for f, t in data if t == 0]

    create_96(cars, bikes)
    create_enhanced(cars, bikes)
    create_resnet(cars, bikes)


if __name__ == '__main__':
    main()
