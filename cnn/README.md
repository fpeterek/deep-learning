# Car/Bike Recognition

## Dataset

In this project, we will be using the following dataset:

https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset


The dataset contains pictures of 2000 motorbikes and 2000 cars. Since the dataset is perfectly
balanced, we can use the accuracy metric. The pictures, at a quick glance, appear to have been
taken in broad daylight, with little to no noise. Thus, we can expect to reach reasonably high
accuracies.

## Preprocessing

First of all, we will need to preprocess the dataset. We will pregenerate the data so as to avoid
doing the preprocessing every time we want to train or test a new model. As part of the
preprocessing process, we will also split the dataset into a training and testing set.

During preprocessing, we create three new datasets. One dataset contains black and white images of
size 96x96 -- with the resizing and converting to grayscale being done by us. This dataset will be
used to train custom models. The second dataset consists of colored images of size 224x224. This
dataset will be used to train the Resnet pre-trained network. The last dataset, once again,
consists of grayscale images of size 96x96, however, this time we will be performing data
enhancement and doubling the size of the training data by rotating the images from the training set.
The last dataset will, yet again, be used to train custom models.

```sh
python3 cnn/preprocess.py
```

## HOG/SVM

The first model we will try is an SVM classifier. As signals for our classifier, we will be using
histograms of oriented gradients. This model only serves as a baseline, so we won't be
experimenting too much here.

```sh
python3 cnn/train.py train-hog --model-name 'svm.model'
```

However, we can see that we actually obtain a reasonably accurate classifier even with such a
simple model. Of course, this is partly given by the simplicity of the dataset.

```
SVM accuracy: 0.941
```

## Custom Small Network

Next, we will try to train our own convolutional network. The architecture of the network
is specified in [cnn/classifiers.py](cnn/classifiers.py). We will also experiment with the size
of the batches and with the learning rate of the model. Defaults for batch size and learning rate
are 64 and 0.001 respectively, for when the parameters are unspecified. In our last attempt, we
will try to train the model on the enhanced dataset.

We will be training over 10 epochs.

```sh
python3 cnn/train.py train-small --model-name 'small-lr001-b64.model' --dataset data/96/
python3 cnn/train.py train-small --model-name 'small-lr001-b16.model' --lr 0.001 --batch-size 16 --dataset data/96/
python3 cnn/train.py train-small --model-name 'small-lr003-b64.model' --lr 0.003 --dataset data/96/
python3 cnn/train.py train-small --model-name 'small-enhanced.model' --lr 0.001 --batch-size 32 --dataset data/enhanced
```

Somewhat surprisingly, the model trained on the enlarged dataset performs the worst, though only
by a little bit. The other three models perform pretty much identically. Lowering batch size had no
effect beyond inflating the duration of the training, and therefore hadn't helped at all.

```
small-lr001-b64.model accuracy: 0.916
small-lr003-b64.model accuracy: 0.919
small-lr001-b16.model accuracy: 0.917
small-enhanced.model accuracy: 0.911
```

## Custom Large Network

In our next experiment, we will train a different, slighly larger network, which we also designed
ourselves. Yet again, we will be experimenting with batch sizes and learning rates, though to a
slightly higher degree and with larger changes to the values, as the changes in the previous
experiment had little to no effect. Lastly, we will also try to train the model on the enlarged
dataset.

Once again, we train all our models over 10 epochs.

```sh
python3 cnn/train.py train-large --model-name 'large-enhanced.model' --lr 0.001 --batch-size 32 --dataset data/enhanced
python3 cnn/train.py train-large --model-name 'large-lr001-b8.model' --lr 0.001 --batch-size 8 --dataset data/96
python3 cnn/train.py train-large --model-name 'large-lr01-b64.model' --lr 0.01 --batch-size 64 --dataset data/96
python3 cnn/train.py train-large --model-name 'large-lr001-b64.model' --lr 0.001 --batch-size 64 --dataset data/96
```

Here, we can see that increasing the learning rate had little effect, however, decreasing batch
size improved the performance by a notable amount. The model trained on the enhanced dataset also
performs a lot better, although it was trained using larger batches.

Finally, we can say that the performance penalty incurred by decreasing batch sizes and enhancing
the dataset proved to be worth the cost.

```
large-lr001-b64.model accuracy: 0.903
large-lr01-b64.model accuracy: 0.905
large-lr001-b8.model accuracy: 0.937
large-enhanced.model accuracy: 0.937
```

## Resnet And Transfer Learning

Finally, we will try to employ transfer learning and adapt an existing pretrained Resnet model to
our needs. This time, we're using batches of size 32 and we're training the model over 8 epochs.

```sh
python3 cnn/train.py train-resnet --model-name 'resnet.model'
```

The network took notably longer to train than the previous models, however, it performs almost
flawlessly with an accuracy just 0.4 percentage points shy of perfection.

```
Resnet accuracy: 0.996
```

## Running The Tests

```sh
python3 cnn
```

