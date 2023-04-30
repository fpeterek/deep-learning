# Recurrent Neural Network

## Dataset

[Trip Advisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)

We will be analyzing the Trip Advisor Hotel Reviews dataset. The dataset contains 20491 records.
Of those records, nearly half are five star reviews. There are also more four star reviews, then
there are one, two and three star reviews combined. Thus, we can say that the dataset is not
perfectly balanced.

The number of occurences of each rating can be seen in the following table.

| Rating | Count |
|--------|-------|
| 1 | 1421 |
| 2 | 1793 |
| 3 | 2184 |
| 4 | 6039 |
| 5 | 9054 |

The dataset only consists of two attributes. The first attribute is the text of the review,
the second attribute is the rating in stars which the hotel received.

## Evaluation Metric

We will use both the F1-score and accuracy to compare the models. We will use accuracy because
it's simple and easy to understand. We will also use F1-score in an attempt to counteract the
imbalance in the training data.

## Preprocessing

### Splitting the Dataset

We will use a simple Python script to split the dataset into multiple parts. We will generate
random numbers using a uniform distribution and randomly select 20 % of the data as the testing
dataset, 10 % of the data as the validation dataset, and the rest will then be reserved for the
training dataset.

### Text Preprocessing

Before attempting to train the models, we must also preprocess the input dataset. Text
preprocessing is done by splitting the input text into words, removing non-ascii characters,
converting all characters to lowercase, removing punctuation, numbers and stopwords. Finally,
we also remove apostrophes from the abbreviations of the word `not`.

### Label Preprocessing

The number 1 must be subtracted from all labels to ensure labels start from zero, not from one.

## Training

For all experiments, we will use the Root Mean Square Propagation optimizer. We will also train
the model over ten epochs using batches of size 64. Fine tuning will only be performed over
three epochs, with the learning rate being set to `0.00001`.

### Custom Embedding

First, we will try to create our own network and train our own word embedding layer.
Our network will consist of an embedding layer, an LSTM layer, a GRU layer, and then a fully
connected network. The sizes of layers and even the number of layers in the FC network vary among
experiments. We will perform three different experiments and train three different models,
starting from a rather small model and trying to make the model bigger with each experiment.

### Transfer Learning

Then, we will try to create a network which utilizes a pretrained embedding layer, though we will
try to fine-tune the embedding layer once the model has been trained. Yet again, we have three
different configurations, starting with a rather small model and attempting to make the model
bigger. We also fine-tune each model.

## Results

In the following table, we can see that all models performed roughly the same. Why that is, I do
not know. Perhaps training the models over more epochs and experimenting with larger models or
more configurations would yield more satisfying results, but I could not get Tensorflow to detect
my GPU.

Or possibly the dataset of around 20k records is just too small to train a reasonably performing
ML model.

All models reached only around 60 % accuracy and an F1-score of 0.5.

| model | accuracy | f1-score |
|-------|----------|----------|
| custom_embedding_1 | 0.6053789731051344 | 0.519446295250419 |
| custom_embedding_2 | 0.5980440097799511 | 0.5147981332721222 |
| custom_embedding_3 | 0.606601466992665 | 0.4981779464937029 |
| glove_embedding_1 | 0.5941320293398533 | 0.49788019028949354 |
| glove_embedding_2 | 0.6014669926650367 | 0.5111293465478799 |
| glove_embedding_3 | 0.5926650366748166 | 0.5077964297721251 |

The results also prompted me to construct a `collections.Counter` over the list of predictions
made by the model, to check whether the models aren't just making random predictions or predicting
only one class over and over again (although the F1-score metric should be able to catch that
case).

```py
print(Counter(pred_y))
```

However, it seems that the distribution of predicted ratings somewhat matches the distribution
of real ratings, the models are just inaccurate and make bad predictions.

```
Counter({4: 2044, 3: 1280, 1: 322, 2: 264, 0: 180})
Counter({4: 1785, 3: 1507, 1: 302, 2: 291, 0: 205})
Counter({4: 1683, 3: 1629, 2: 472, 1: 190, 0: 116})
Counter({4: 2197, 3: 1029, 2: 327, 1: 299, 0: 238})
Counter({4: 2185, 3: 1066, 2: 305, 1: 270, 0: 264})
Counter({4: 2171, 3: 1115, 2: 278, 0: 267, 1: 259})
```
