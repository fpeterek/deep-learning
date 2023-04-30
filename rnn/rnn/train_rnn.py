import tensorflow as tf
import keras


def save_callback():
    return tf.keras.callbacks.ModelCheckpoint(
        filepath='weights.best.tf',
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)


def train_custom(model, train_x, train_y, valid_x, valid_y):

    conf = {
            'batch_size': 64,
            'epochs': 10,
            'callbacks': [save_callback()],
            'validation_data': (valid_x.values, valid_y.values),
            'verbose': 0,
            }

    model.fit(train_x.values, train_y.values, **conf)

    model.load_weights('weights.best.tf')


def train_transfer_learning(model, emb, train_x, train_y, valid_x, valid_y):

    conf = {
            'batch_size': 64,
            'epochs': 10,
            'callbacks': [save_callback()],
            'validation_data': (valid_x.values, valid_y.values),
            'verbose': 0,
            }

    model.fit(train_x.values, train_y.values, **conf)

    model.load_weights('weights.best.tf')

    emb.trainable = True

    model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.00001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    conf = {
            'batch_size': 64,
            'epochs': 3,
            'callbacks': [save_callback()],
            'validation_data': (valid_x.values, valid_y.values),
            'verbose': 0,
            }

    model.fit(train_x.values, train_y.values, **conf)

    model.load_weights('weights.best.tf')
