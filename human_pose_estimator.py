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
import os
import random
from model import Model
from glob import glob
from time import time
from enum import Enum, auto

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tqdm import tqdm
from generator import DataGenerator
from lr_scheduler import LRScheduler


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Limb(Enum):
    HEAD = 0
    NECK = auto()
    RIGHT_SHOULDER = auto()
    RIGHT_ELBOW = auto()
    RIGHT_WRIST = auto()
    LEFT_SHOULDER = auto()
    LEFT_ELBOW = auto()
    LEFT_WRIST = auto()
    RIGHT_HIP = auto()
    RIGHT_KNEE = auto()
    RIGHT_ANKLE = auto()
    LEFT_HIP = auto()
    LEFT_KNEE = auto()
    LEFT_ANKLE = auto()


class HumanPoseEstimator:
    def __init__(
            self,
            train_image_path,
            input_shape,
            lr,
            momentum,
            batch_size,
            iterations,
            decay=5e-4,
            training_view=False,
            pretrained_model_path='',
            validation_image_path='',
            output_tensor_dimension=2,
            confidence_threshold=0.25,
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
        self.output_tensor_dimension = output_tensor_dimension
        self.confidence_threshold = confidence_threshold
        self.img_type = cv2.IMREAD_COLOR
        self.live_view_time = time()
        assert self.output_tensor_dimension in [1, 2]
        if input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE

        self.limb_size = len(Limb)
        if self.output_tensor_dimension == 1:
            self.output_size = self.limb_size * 3
        elif self.output_tensor_dimension == 2:
            self.output_size = self.limb_size + 3
        if pretrained_model_path == '':
            self.model = Model(input_shape=self.input_shape, output_size=self.output_size, decay=decay).build(output_tensor_dimension=self.output_tensor_dimension)
            self.model.save('model.h5', include_optimizer=False)
        else:
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)
        self.output_shape = self.model.output_shape[1:]

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
            output_shape=self.output_shape,
            output_tensor_dimension=self.output_tensor_dimension,
            batch_size=self.batch_size,
            limb_size=self.limb_size)
        self.validation_data_generator = DataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            output_tensor_dimension=self.output_tensor_dimension,
            batch_size=self.batch_size,
            limb_size=self.limb_size)
        self.lr_scheduler = LRScheduler(iterations=self.iterations, lr=self.lr)

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

    def calculate_pck(self, dataset='validation', distance_threshold=0.1):  # PCK : percentage of correct keypoints, the metric of keypoints detection model
        assert dataset in ['train', 'validation']
        total_count = 0
        correct_count = 0
        image_paths = self.train_image_paths if dataset == 'train' else self.validation_image_paths

        np.random.shuffle(image_paths)
        image_paths = image_paths[:2000]

        head_neck_distance_count = 0
        head_neck_distance_sum = 0.0
        for image_path in tqdm(image_paths):
            img, path = DataGenerator.load_img(image_path, self.input_shape[-1] == 3)
            img = DataGenerator.resize(img, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(img).reshape((1,) + self.input_shape).astype('float32') / 255.0
            y_pred = np.asarray(self.graph_forward(self.model, x)).reshape(self.output_shape).astype('float32')
            y_pred = self.post_process(y_pred)
            
            with open(f'{path[:-4]}.txt', 'rt') as f:
                lines = f.readlines()
            y_true = np.zeros(shape=(self.limb_size, 3), dtype=np.float32)
            for i, line in enumerate(lines):
                if i == self.limb_size:
                    break
                confidence, x_pos, y_pos = list(map(float, line.split()))
                y_true[i][0] = confidence
                y_true[i][1] = x_pos
                y_true[i][2] = y_pos
                if confidence == 1.0:
                    total_count += 1

            if y_true[0][0] == 1.0 and y_true[1][0] == 1.0:
                x_pos_head = y_true[0][1]
                y_pos_head = y_true[0][2]
                x_pos_neck = y_true[1][1]
                y_pos_neck = y_true[1][2]
                distance = np.sqrt(np.square(x_pos_head - x_pos_neck) + np.square(y_pos_head - y_pos_neck))
                head_neck_distance_sum += distance
                head_neck_distance_count += 1

            for i in range(self.limb_size):
                if y_pred[i][0] < self.confidence_threshold:
                    continue
                x_pos_true = y_true[i][1]
                y_pos_true = y_true[i][2]
                x_pos_pred = y_pred[i][1]
                y_pos_pred = y_pred[i][2]
                distance = np.sqrt(np.square(x_pos_true - x_pos_pred) + np.square(y_pos_true - y_pos_pred))
                if distance < distance_threshold:
                    correct_count += 1

        head_neck_distance = head_neck_distance_sum / float(head_neck_distance_count)
        print(f'head neck distance : {head_neck_distance:.4f}')

        pck = correct_count / float(total_count)
        print(f'{dataset} data PCK@{int(distance_threshold * 100)} : {pck:.4f}')
        return pck

    def compute_gradient_1d(self, model, optimizer, x, y_true, limb_size):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.dtypes.int32)
            y_true = tf.reshape(y_true, (batch_size, limb_size, 3))
            y_pred = tf.reshape(y_pred, (batch_size, limb_size, 3))
            confidence_loss = tf.reduce_sum(tf.reduce_mean(K.binary_crossentropy(y_true[:, :, 0], y_pred[:, :, 0]), axis=0))
            regression_loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(-tf.math.log(1.0 - tf.abs(y_true[:, :, 1:] - y_pred[:, :, 1:])), axis=-1) * y_true[:, :, 0], axis=0))
            loss = confidence_loss + regression_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def compute_gradient_2d(self, model, optimizer, x, y_true, limb_size):
        def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
            p_t = tf.where(K.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
            alpha_factor = K.ones_like(y_true) * alpha
            alpha_t = tf.where(K.equal(y_true, 1.0), alpha_factor, 1.0 - alpha_factor)
            cross_entropy = K.binary_crossentropy(y_true, y_pred)
            weight = alpha_t * K.pow((1.0 - p_t), gamma)
            loss = weight * cross_entropy
            return loss

        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            confidence_true = y_true[:, :, :, 0]
            confidence_pred = y_pred[:, :, :, 0]
            confidence_loss = tf.reduce_sum(tf.reduce_mean(focal_loss(confidence_true, confidence_pred), axis=0))
            regression_true = y_true[:, :, :, 1:3]
            regression_pred = y_pred[:, :, :, 1:3]
            regression_loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(-tf.math.log((1.0 + K.epsilon()) - tf.abs(regression_true - regression_pred)), axis=-1) * confidence_true, axis=0))
            class_true = y_true[:, :, :, 3:]
            class_pred = y_pred[:, :, :, 3:]
            classification_loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(class_true, class_pred), axis=-1) * confidence_true, axis=0))
            loss = confidence_loss + classification_loss + (regression_loss * 5.0)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def fit(self):
        optimizer = tf.keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, nesterov=True)
        # optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum)
        # optimizer = tf.keras.optimizers.RMSprop(lr=self.lr)
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples')
        iteration_count = 0
        min_val_loss = 999999999.0
        os.makedirs('checkpoints', exist_ok=True)
        if self.output_tensor_dimension == 1:
            compute_gradient = tf.function(self.compute_gradient_1d)
        elif self.output_tensor_dimension == 2:
            compute_gradient = tf.function(self.compute_gradient_2d)
        for x, y_true in self.train_data_generator.flow():
            self.lr_scheduler.schedule_one_cycle(optimizer, iteration_count)
            loss = compute_gradient(self.model, optimizer, x, y_true, self.limb_size)
            iteration_count += 1
            if self.training_view_flag:
                self.training_view_function()
            print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
            if iteration_count % 10000 == 0:
                print()
                self.model.save(f'checkpoints/model_{iteration_count}_iter.h5', include_optimizer=False)
                # val_loss = self.evaluate(self.model, self.validation_data_generator, loss_fn)
                val_loss = 0.0
                print(f'val_loss : {val_loss:.4f}')
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    self.model.save(f'checkpoints/model_{iteration_count}_iter_{val_loss:.4f}_val_loss.h5', include_optimizer=False)
                    print('minimum val loss model saved')
                print()
            if iteration_count == self.iterations:
                print('\ntrain end successfully')
                return

    def predict_images(self, dataset='validation'):
        assert dataset in ['train', 'validation']
        if dataset == 'train':
            image_paths = self.train_image_paths
        elif dataset == 'validation':
            image_paths = self.validation_image_paths
        for img_path in image_paths:
            img = self.predict(DataGenerator.load_img(img_path, color=True)[0])
            cv2.imshow(f'{dataset} images', img)
            key = cv2.waitKey(0)
            if key == 27:
                break

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def circle_if_valid(self, img, v):
        if v[0] > self.confidence_threshold:
            x = int(v[1] * img.shape[1])
            y = int(v[2] * img.shape[0])
            img = cv2.circle(img, (x, y), 6, (128, 255, 128), thickness=-1, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (x, y), 3, (32, 32, 192), thickness=-1, lineType=cv2.LINE_AA)
        return img

    def line_if_valid(self, img, p1, p2):
        if p1[0] > self.confidence_threshold and p2[0] > self.confidence_threshold:
            x1 = int(p1[1] * img.shape[1])
            y1 = int(p1[2] * img.shape[0])
            x2 = int(p2[1] * img.shape[1])
            y2 = int(p2[2] * img.shape[0])
            img = cv2.line(img, (x1, y1), (x2, y2), (64, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        return img

    def draw_skeleton(self, img, y):
        img = self.line_if_valid(img, y[Limb.HEAD.value], y[Limb.NECK.value])

        img = self.line_if_valid(img, y[Limb.NECK.value], y[Limb.RIGHT_SHOULDER.value])
        img = self.line_if_valid(img, y[Limb.RIGHT_SHOULDER.value], y[Limb.RIGHT_ELBOW.value])
        img = self.line_if_valid(img, y[Limb.RIGHT_ELBOW.value], y[Limb.RIGHT_WRIST.value])

        img = self.line_if_valid(img, y[Limb.NECK.value], y[Limb.LEFT_SHOULDER.value])
        img = self.line_if_valid(img, y[Limb.LEFT_SHOULDER.value], y[Limb.LEFT_ELBOW.value])
        img = self.line_if_valid(img, y[Limb.LEFT_ELBOW.value], y[Limb.LEFT_WRIST.value])

        img = self.line_if_valid(img, y[Limb.RIGHT_HIP.value], y[Limb.LEFT_HIP.value])

        img = self.line_if_valid(img, y[Limb.RIGHT_SHOULDER.value], y[Limb.RIGHT_HIP.value])
        img = self.line_if_valid(img, y[Limb.RIGHT_HIP.value], y[Limb.RIGHT_KNEE.value])
        img = self.line_if_valid(img, y[Limb.RIGHT_KNEE.value], y[Limb.RIGHT_ANKLE.value])

        img = self.line_if_valid(img, y[Limb.LEFT_SHOULDER.value], y[Limb.LEFT_HIP.value])
        img = self.line_if_valid(img, y[Limb.LEFT_HIP.value], y[Limb.LEFT_KNEE.value])
        img = self.line_if_valid(img, y[Limb.LEFT_KNEE.value], y[Limb.LEFT_ANKLE.value])
        for v in y:
            img = self.circle_if_valid(img, v)
        return img

    def post_process(self, y):
        target_shape = (self.limb_size, 3)
        if self.output_tensor_dimension == 1:
            return y.reshape(target_shape)
        else:
            rows, cols = self.output_shape[:2]
            res = np.zeros(shape=(self.limb_size, 3), dtype=np.float32)
            for row in range(rows):
                for col in range(cols):
                    for i in range(self.limb_size):
                        confidence = y[row][col][0]
                        if confidence < self.confidence_threshold:
                            continue
                        class_index = 0
                        max_class_score = 0.0
                        for j in range(self.limb_size):
                            class_score = y[row][col][j+3]
                            if class_score > max_class_score:
                                max_class_score = class_score
                                class_index = j
                        confidence *= max_class_score
                        if confidence < self.confidence_threshold:
                            continue
                        if confidence > res[class_index][0]:
                            x_pos = y[row][col][1]
                            y_pos = y[row][col][2]
                            x_pos = (col + x_pos) / float(cols)
                            y_pos = (row + y_pos) / float(rows)
                            res[class_index][0] = confidence
                            res[class_index][1] = x_pos
                            res[class_index][2] = y_pos
            return res

    def predict(self, color_img, view_size=(256, 512)):
        raw = color_img
        if self.input_shape[-1] == 1:
            img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
        else:
            img = color_img
        x = np.asarray(DataGenerator.resize(img, (self.input_shape[1], self.input_shape[0]))).reshape((1,) + self.input_shape).astype('float32') / 255.0
        y = np.asarray(self.graph_forward(self.model, x)).reshape(self.output_shape)
        y = self.post_process(y)
        img = self.draw_skeleton(DataGenerator.resize(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), view_size), y)
        return img

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_time < 0.5:
            return
        self.live_view_time = cur_time
        train_image = self.predict(DataGenerator.load_img(np.random.choice(self.train_image_paths), color=True)[0])
        validation_image = self.predict(DataGenerator.load_img(np.random.choice(self.validation_image_paths), color=True)[0])
        cv2.imshow('train', train_image)
        cv2.imshow('validation', validation_image)
        cv2.waitKey(1)
