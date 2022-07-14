"""
Authors : inzapp

Github url : https://github.com/inzapp/human-pose-estimator

Copyright 2022 inzapp Authors. All Rights Reserved.

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
from cv2 import cv2
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import tensorflow as tf


class DataGenerator:
    def __init__(self, image_paths, input_shape, output_shape, output_tensor_dimension, batch_size, limb_size):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_tensor_dimension = output_tensor_dimension
        self.batch_size = batch_size
        self.limb_size = limb_size
        self.pool = ThreadPoolExecutor(8)
        self.image_index = 0

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        fs = []
        for i in range(self.batch_size):
            fs.append(self.pool.submit(self.load_img, self.get_next_image_path(), self.input_shape[-1]))
        for f in fs:
            x, path = f.result()
            x = self.resize(x, (self.input_shape[1], self.input_shape[0]))
            x = self.random_blur(x)
            x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
            batch_x.append(x)

            label_path = f'{path[:-4]}.txt'
            with open(label_path, 'rt') as file:
                lines = file.readlines()
            if self.output_tensor_dimension == 1:
                y = []
            elif self.output_tensor_dimension == 2:
                y = np.zeros(shape=self.output_shape, dtype=np.float32)
            for i, line in enumerate(lines):
                if i == self.limb_size:
                    break
                confidence, x_pos, y_pos = list(map(float, line.split()))
                if self.output_tensor_dimension == 1:
                    y += [confidence, x_pos, y_pos]
                elif self.output_tensor_dimension == 2:
                    x_pos, y_pos = np.clip([x_pos, y_pos], 0.0, 1.0 - 1e-4)
                    output_rows = self.output_shape[0]
                    output_cols = self.output_shape[1]
                    row = int(y_pos * output_rows)
                    col = int(x_pos * output_cols)
                    y[row][col][0] = confidence
                    y[row][col][1] = (x_pos - float(col) / output_cols) / (1.0 / output_cols)
                    y[row][col][2] = (y_pos - float(row) / output_rows) / (1.0 / output_rows)
                    y[row][col][i+3] = 1.0
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size,) + self.output_shape).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    @staticmethod
    def resize(img, size):
        if size[0] > img.shape[1] or size[1] > img.shape[0]:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def get_next_image_path(self):
        path = self.image_paths[self.image_index]
        self.image_index += 1
        if self.image_index == len(self.image_paths):
            np.random.shuffle(self.image_paths)
            self.image_index = 0
        return path

    def random_blur(self, img):
        if np.random.uniform() > 0.5:
            kernel_size = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img

    @staticmethod
    def load_img(path, channels):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        return img, path
