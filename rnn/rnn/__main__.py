import sys
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from preprocess import preprocess
import create_rnn
import train_rnn


def load_ds(path: str):
    return preprocess(pd.read_csv(path, sep=',', header=0))


def test_model(model_creator, train_x, train_y,
               valid_x, valid_y, test_x, test_y):

    print(model_creator.__name__)

    rnn = model_creator(train_x)
    train_rnn.train_custom(rnn, train_x, train_y, valid_x, valid_y)

    pred_y = np.argmax(rnn.predict(test_x), axis=1)

    print(Counter(pred_y))

    acc = accuracy_score(y_true=test_y, y_pred=pred_y)
    f1 = f1_score(y_true=test_y, y_pred=pred_y, average='macro')

    return model_creator.__name__, acc, f1


def test_transfer_learning(model_creator, train_x, train_y,
                           valid_x, valid_y, test_x, test_y):

    print(model_creator.__name__)

    rnn, emb = model_creator(train_x)
    train_rnn.train_transfer_learning(rnn, emb, train_x,
                                      train_y, valid_x, valid_y)

    pred_y = np.argmax(rnn.predict(test_x), axis=1)

    print(Counter(pred_y))

    acc = accuracy_score(y_true=test_y, y_pred=pred_y)
    f1 = f1_score(y_true=test_y, y_pred=pred_y, average='macro')

    return model_creator.__name__, acc, f1


def main(train: str, valid: str, test: str):
    train = load_ds(train)
    valid = load_ds(valid)
    test = load_ds(test)

    train_x = train['ReviewString']
    train_y = train['Rating']
    valid_x = valid['ReviewString']
    valid_y = valid['Rating']
    test_x = test['ReviewString']
    test_y = test['Rating']

    print(train.head())

    results = dict()

    custom = [
            create_rnn.custom_embedding_1, create_rnn.custom_embedding_2,
            create_rnn.custom_embedding_3]

    for fn in custom:
        model, acc, f1 = test_model(
                fn, train_x, train_y, valid_x, valid_y, test_x, test_y)

        results[model] = (acc, f1)

    glove = [
            create_rnn.glove_embedding_1, create_rnn.glove_embedding_2,
            create_rnn.glove_embedding_3]

    for fn in glove:
        model, acc, f1 = test_transfer_learning(
                fn, train_x, train_y, valid_x, valid_y, test_x, test_y)

        results[model] = (acc, f1)

    print(results)

    print('| model | accuracy | f1-score |')
    print('-------------------------------')
    for m in results:
        acc, f1 = results[m]
        print(f'| {m} | {acc} | {f1} |')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
