
"""
Authors : inzapp

Github url : https://github.com/inzapp/human-pose-estimator

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


class Model:
    def __init__(self, input_shape, output_size, decay):
        self.input_shape = input_shape
        self.output_size = output_size
        self.decay = decay

    def build(self, output_tensor_dimension):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = input_layer
        x = self.conv(x, filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        x = self.max_pool(x)

        x = self.conv(x, filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        x = self.conv(x, filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        x = self.max_pool(x)

        x = self.conv(x, filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        x = self.conv(x, filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        x = self.spatial_attention(x)
        x = self.max_pool(x)

        x = self.conv(x, filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        x = self.conv(x, filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        f0 = x
        x = self.max_pool(x)

        x = self.conv(x, filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        x = self.conv(x, filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        f1 = x
        x = self.max_pool(x)

        x = self.conv(x, filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        x = self.conv(x, filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu')
        f2 = x

        x = self.feature_pyramid_network([f0, f1, f2], [128, 128, 256], kernel_size=3, kernel_initializer='he_normal', activation='relu')
        if output_tensor_dimension == 1:
            x = self.conv(x, filters=self.output_size, kernel_size=1, kernel_initializer='he_normal', activation='relu')
            x = tf.keras.layers.Flatten()(x)
            x = self.dense(x, units=self.output_size, kernel_initializer='glorot_normal', activation='sigmoid')
        elif output_tensor_dimension == 2:
            x = self.conv(x, filters=self.output_size, kernel_size=1, kernel_initializer='glorot_normal', activation='sigmoid')
        return tf.keras.models.Model(input_layer, x)

    def conv(self, x, filters, kernel_size, kernel_initializer, activation, bn=False):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.decay) if self.decay > 0.0 else None,
            use_bias=False if bn else True,
            padding='same')(x)
        if bn:
            x = self.bn(x)
        x = self.activation(x, activation=activation)
        return x

    def dense(self, x, units, kernel_initializer, activation, bn=False):
        x = tf.keras.layers.Dense(
            units=units,
            kernel_initializer=kernel_initializer,
            use_bias=False if bn else True)(x)
        if bn:
            x = self.bn(x)
        x = self.activation(x, activation=activation)
        return x

    def activation(self, x, activation, alpha=0.1):
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=alpha)(x) 
        else:
            x = tf.keras.layers.Activation(activation)(x)
        return x

    def spatial_attention(self, x):
        input_layer = x
        input_filters = input_layer.shape[-1]
        squeezed_filters = input_filters // 16
        if squeezed_filters <= 2:
            squeezed_filters = 2
        x = self.conv(x, filters=squeezed_filters, kernel_size=1, kernel_initializer='he_normal', activation='relu')
        x = self.conv(x, filters=squeezed_filters, kernel_size=7, kernel_initializer='he_normal', activation='relu')
        x = tf.keras.layers.Conv2D(
            filters=input_filters,
            kernel_size=1,
            kernel_initializer='glorot_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2=self.decay) if self.decay > 0.0 else None,
            bias_initializer='zeros',
            use_bias=False,
            padding='same')(x)
        x = self.activation(x, activation='sigmoid')
        return self.multiply([input_layer, x])

    def feature_pyramid_network(self, layers, filters, kernel_size, kernel_initializer, activation, bn=False, return_layers=False):
        layers = list(reversed(layers))
        if type(filters) == list:
            filters = list(reversed(filters))
        for i in range(len(layers)):
            layers[i] = self.conv(layers[i], filters=filters if type(filters) == int else filters[i], kernel_size=1, kernel_initializer=kernel_initializer, bn=bn, activation=activation)
        ret = []
        if return_layers:
            ret.append(layers[0])
        for i in range(len(layers) - 1):
            x = tf.keras.layers.UpSampling2D()(layers[i] if i == 0 else x)
            if type(filters) == list and filters[i] != filters[i + 1]:
                x = self.conv(x, filters=filters[i + 1], kernel_size=1, kernel_initializer=kernel_initializer, bn=bn, activation=activation)
            x = self.add([x, layers[i + 1]])
            x = self.conv(x, filters=filters if type(filters) == int else filters[i + 1], kernel_size=3, kernel_initializer=kernel_initializer, bn=bn, activation=activation)
            if return_layers:
                ret.append(x)
        return list(reversed(ret)) if return_layers else x

    def max_pool(self, x):
        return tf.keras.layers.MaxPool2D()(x)

    def bn(self, x):
        return tf.keras.layers.BatchNormalization()(x)

    def add(self, layers):
        return tf.keras.layers.Add()(layers)

    def multiply(self, layers):
        return tf.keras.layers.Multiply()(layers)
