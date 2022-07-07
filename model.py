"""
Authors : inzapp

Github url : https://github.com/inzapp/sigmoid-classifier

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tensorflow as tf


def get_model(input_shape, output_node_size):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        kernel_initializer='he_normal',
        padding='same',
        use_bias=True)(input_layer)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        kernel_initializer='he_normal',
        padding='same',
        use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        kernel_initializer='he_normal',
        padding='same',
        use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        kernel_initializer='he_normal',
        padding='same',
        use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        kernel_initializer='he_normal',
        padding='same',
        use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        kernel_initializer='he_normal',
        padding='same',
        use_bias=True)(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters=output_node_size,
        kernel_size=1,
        kernel_initializer='glorot_normal',
        activation='sigmoid')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='output')(x)
    return tf.keras.models.Model(input_layer, x)
