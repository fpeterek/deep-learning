import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, LSTM, GRU


def in_emb_layers(words):
    embed_dim = 128
    vocab_size = 10_000
    seq_len = 30

    input_layer = keras.layers.Input(shape=(1,), dtype=tf.string)

    vectorization = TextVectorization(max_tokens=vocab_size,
                                      output_mode='int',
                                      output_sequence_length=seq_len)
    vectorization.adapt(words)
    vec = vectorization(input_layer)
    emb = keras.layers.Embedding(vocab_size, embed_dim)(vec)

    return input_layer, emb


def glove_im_emb(words):
    embed_dim = 50
    vocab_size = 10_000
    seq_len = 20

    path_to_glove_file = 'glove/glove.6B.50d.txt'

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    input_layer = keras.layers.Input(shape=(1,), dtype=tf.string)

    vectorization = TextVectorization(max_tokens=vocab_size,
                                      output_mode='int',
                                      output_sequence_length=seq_len)

    vectorization.adapt(words)

    voc = vectorization.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    num_tokens = len(voc) + 2
    hits = 0
    misses = 0

    embedding_matrix = np.zeros((num_tokens, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    embed_init = keras.initializers.Constant(embedding_matrix)

    vec = vectorization(input_layer)

    emb = keras.layers.Embedding(
            num_tokens,
            embed_dim,
            embeddings_initializer=embed_init,
            trainable=False)(vec)

    return input_layer, emb


def custom_embedding_1(words):
    input, emb = in_emb_layers(words)

    x = LSTM(64, activation='relu', return_sequences=True)(emb)
    x = GRU(64, activation='relu', return_sequences=True)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, 'relu')(x)
    x = keras.layers.Dense(32, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(5, 'softmax')(x)

    model = keras.Model(input, output)

    model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    return model


def custom_embedding_2(words):
    input, emb = in_emb_layers(words)

    x = LSTM(128, activation='relu', return_sequences=True)(emb)
    x = GRU(128, activation='relu', return_sequences=True)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(5, 'softmax')(x)

    model = keras.Model(input, output)

    model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    return model


def custom_embedding_3(words):
    input, emb = in_emb_layers(words)

    x = LSTM(256, activation='relu', return_sequences=True)(emb)
    x = GRU(256, activation='relu', return_sequences=True)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(5, 'softmax')(x)

    model = keras.Model(input, output)

    model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    return model


def glove_embedding_1(words):
    input, emb = glove_im_emb(words)

    x = LSTM(64, activation='relu', return_sequences=True)(emb)
    x = GRU(64, activation='relu', return_sequences=True)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, 'relu')(x)
    x = keras.layers.Dense(32, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(5, 'softmax')(x)

    model = keras.Model(input, output)

    model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    return model, emb


def glove_embedding_2(words):
    input, emb = glove_im_emb(words)

    x = LSTM(128, activation='relu', return_sequences=True)(emb)
    x = GRU(128, activation='relu', return_sequences=True)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, 'relu')(x)
    x = keras.layers.Dense(128, 'relu')(x)
    x = keras.layers.Dense(64, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(5, 'softmax')(x)

    model = keras.Model(input, output)

    model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    return model, emb


def glove_embedding_3(words):
    input, emb = glove_im_emb(words)

    x = LSTM(128, activation='relu', return_sequences=True)(emb)
    x = GRU(128, activation='relu', return_sequences=True)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, 'relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, 'relu')(x)
    output = keras.layers.Dense(5, 'softmax')(x)

    model = keras.Model(input, output)

    model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    return model, emb
