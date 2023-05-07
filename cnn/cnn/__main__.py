import warnings

import cv2 as cv

import util
from classifiers import SmallCNNClassifier, LargeCNNClassifier
from classifiers import ResnetClassifier, HOGClassifier


warnings.filterwarnings(action='ignore', category=UserWarning)


def load_test_imgs(base_path):
    ds = util.load_ds(f'{base_path.strip("/")}/test')

    return [(cv.imread(img), label) for img, label in ds]


def test_classifier(classifier, ds):
    correct = 0

    for img, label in ds:
        pred = classifier(img)
        correct += (pred == label)

    return correct / len(ds)


def test_resnet():
    classifier = ResnetClassifier('resnet.model')
    ds = load_test_imgs('data/resnet/')
    acc = test_classifier(classifier, ds)

    print(f'Resnet accuracy: {acc:.3f}')


def test_svm():
    classifier = HOGClassifier('svm.model')
    ds = load_test_imgs('data/96/')
    acc = test_classifier(classifier, ds)

    print(f'SVM accuracy: {acc:.3f}')


def test_one(model_name, model_class, ds):
    classifier = model_class(model_name)
    acc = test_classifier(classifier, ds)
    print(f'{model_name} accuracy: {acc:.3f}')


def test_small():
    ds = load_test_imgs('data/96')

    test_one('small-lr001-b64.model', SmallCNNClassifier, ds)
    test_one('small-lr003-b64.model', SmallCNNClassifier, ds)
    test_one('small-lr001-b16.model', SmallCNNClassifier, ds)

    ds = load_test_imgs('data/enhanced/')
    test_one('small-enhanced.model', SmallCNNClassifier, ds)


def test_large():
    ds = load_test_imgs('data/96')

    test_one('large-lr001-b64.model', LargeCNNClassifier, ds)
    test_one('large-lr01-b64.model', LargeCNNClassifier, ds)
    test_one('large-lr001-b8.model', LargeCNNClassifier, ds)

    ds = load_test_imgs('data/enhanced/')
    test_one('large-enhanced.model', LargeCNNClassifier, ds)


def main():
    test_svm()
    test_resnet()
    test_small()
    test_large()


if __name__ == '__main__':
    main()
