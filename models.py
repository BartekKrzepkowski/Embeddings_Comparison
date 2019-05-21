# https://github.com/tensorflow/tensorboard/tree/master/docs/r2
# https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Dense, GRU, Bidirectional, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, MaxPooling1D, Reshape, concatenate, Conv2D, MaxPool2D, Concatenate, Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf


def model_rnn(kwargs):
    K.clear_session()
    nn_input = Input(shape=kwargs["shape"], dtype=kwargs["input_dtype"])
    
    x = kwargs["emb_layer"](nn_input)
    x = SpatialDropout1D(0.1)(x)
    
    x = Bidirectional(GRU(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))(x)
    x = Bidirectional(GRU(32, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.1)(x)
    
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    
    nn_pred = Dense(kwargs["out_units"], activation=kwargs["out_activation"])(x)
    
    model = Model(inputs=nn_input, outputs=nn_pred)
    
    model.compile(
        loss=kwargs["loss"],
        optimizer=kwargs["optimizer"],
        metrics=["accuracy"]
    )
    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return model


def model_ffnn(kwargs):
    K.clear_session()
    if kwargs["is_bert"]:
        nn_input = kwargs["bert_input_layer"]
    else:
        nn_input = Input(shape=kwargs["shape"], dtype=kwargs["input_dtype"])
    
    x = kwargs["emb_layer"](nn_input)
    
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    
    nn_pred = Dense(kwargs["out_units"], activation=kwargs["out_activation"])(x)
    
    model = Model(inputs=nn_input, outputs=nn_pred)
    
    model.compile(
        loss=kwargs["loss"],
        optimizer=kwargs["optimizer"],
        metrics=["accuracy"]
    )
    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return model


# Architektura Yoon Kima z dynamicznym wejściem
def model_cnn(kwargs):
    K.clear_session()
    nn_input = Input(shape=kwargs["shape"], dtype=kwargs["input_dtype"])
    
    x = kwargs["emb_layer"](nn_input)
    x = SpatialDropout1D(0.1)(x)
    x = Reshape((-1, 1024, 1), input_shape=K.int_shape(x))(x)
    
    maxpool_pool = []
    filter_sizes = [1, 2, 3, 5]
    for i in range(len(filter_sizes)):
        conv = Conv2D(32, kernel_size=(filter_sizes[i], 1024),
                                     kernel_initializer='he_normal', activation='relu')(x)
        conv = Reshape((-1, K.int_shape(conv)[3]))(conv)
        global_pool = GlobalAveragePooling1D()(conv)
        maxpool_pool.append(global_pool)
        
    x = Concatenate(axis=1)(maxpool_pool)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    
    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    
    nn_pred = Dense(kwargs["out_units"], activation=kwargs["out_activation"])(x)
    
    model = Model(inputs=nn_input, outputs=nn_pred)
    
    model.compile(
        loss=kwargs["loss"],
        optimizer=kwargs["optimizer"],
        metrics=["accuracy"]
    )
    sess = K.get_session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    return model





