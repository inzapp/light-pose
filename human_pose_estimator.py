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
import os
import random
from glob import glob
from time import time

import cv2
import tensorflow as tf

from tqdm import tqdm
from model import get_model
from generator import DataGenerator


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
live_view_previous_time = time()


class HumanPoseEstimator:
    def __init__(
            self,
            train_image_path,
            input_shape,
            lr,
            momentum,
            batch_size,
            iterations,
            training_view=False,
            pretrained_model_path='',
            validation_image_path='',
            validation_split=0.2):
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.validation_split = validation_split
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.training_view_flag = training_view
        self.img_type = cv2.IMREAD_COLOR
        if input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE

        self.output_node_size = 16 * 3
        if pretrained_model_path == '':
            self.model = get_model(self.input_shape, output_node_size=self.output_node_size)
        else:
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)

        self.train_image_paths = list()
        self.validation_image_paths = list()
        if self.validation_image_path != '':
            self.train_image_paths, _ = self.init_image_paths(self.train_image_path)
            self.validation_image_paths, _ = self.init_image_paths(self.validation_image_path)
        elif self.validation_split > 0.0:
            self.train_image_paths, self.validation_image_paths = self.init_image_paths(self.train_image_path, self.validation_split)

        self.train_data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            output_node_size=self.output_node_size)
        self.validation_data_generator = DataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            output_node_size=self.output_node_size)

    @staticmethod
    def init_image_paths(image_path, validation_split=0.0):
        all_image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        random.shuffle(all_image_paths)
        num_cur_class_train_images = int(len(all_image_paths) * (1.0 - validation_split))
        image_paths = all_image_paths[:num_cur_class_train_images]
        validation_image_paths = all_image_paths[num_cur_class_train_images:]
        return image_paths, validation_image_paths

    def evaluate(self, model, generator_flow, loss_fn):
        loss_sum = 0.0
        for batch_x, batch_y in tqdm(generator_flow):
            y_pred = model(batch_x, training=False)
            loss_sum += tf.reduce_mean(np.square(batch_y - y_pred))
        return loss_sum / tf.cast(len(generator_flow), dtype=tf.float32) 

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mse

    def fit(self):
        optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum)
        # optimizer = tf.keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, nesterov=True)
        self.model.summary()

        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples')

        iteration_count = 0
        min_val_loss = 999999999.0
        os.makedirs('checkpoints', exist_ok=True)
        while True:
            for x, y_true in self.train_data_generator.flow():
                loss = self.compute_gradient(self.model, optimizer, x, y_true)
                iteration_count += 1
                if self.training_view_flag:
                    pass
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                if iteration_count % 10000 == 0:
                    print()
                    # val_loss = self.evaluate(self.model, self.validation_data_generator.flow(), loss_fn)
                    val_loss = 0.0
                    print(f'val_loss : {val_loss:.4f}')
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        self.model.save(f'checkpoints/model_{iteration_count}_iter_{val_loss:.4f}_val_loss.h5', include_optimizer=False)
                        print('minimum val loss model saved')
                    print()
                if iteration_count == self.iterations:
                    print('train end successfully')
                    exit(0)

    def predict_validation_images(self):
        for img_path in self.validation_image_paths:
            pass
