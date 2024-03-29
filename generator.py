"""
Authors : inzapp

Github url : https://github.com/inzapp/light-pose

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
import cv2
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import albumentations as A


class DataGenerator:
    def __init__(self, image_paths, input_shape, output_shape, output_tensor_dimension, batch_size, limb_size):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_tensor_dimension = output_tensor_dimension
        self.batch_size = batch_size
        self.limb_size = limb_size
        self.image_index = 0
        self.pool = ThreadPoolExecutor(8)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.3),
            A.GaussianBlur(p=0.5, blur_limit=(5, 5))
        ])

    def flow(self):
        while True:
            fs = []
            for i in range(self.batch_size):
                fs.append(self.pool.submit(self.load_img, self.get_next_image_path(), color=self.input_shape[-1] == 3))
            batch_x = []
            batch_y = []
            for f in fs:
                img, path = f.result()
                img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
                img = self.transform(image=img)['image']
                x = np.asarray(img).reshape(self.input_shape).astype('float32') / 255.0
                batch_x.append(x)

                with open(f'{path[:-4]}.txt', 'rt') as file:
                    lines = file.readlines()
                y = np.zeros(shape=self.output_shape, dtype=np.float32)
                for i, line in enumerate(lines):
                    if i == self.limb_size:
                        break
                    confidence, x_pos, y_pos = list(map(float, line.split()))
                    if self.output_tensor_dimension == 1:
                        y[i*3+0] = confidence
                        y[i*3+1] = x_pos 
                        y[i*3+2] = y_pos 
                    elif self.output_tensor_dimension == 2:
                        output_rows = self.output_shape[0]
                        output_cols = self.output_shape[1]
                        x_pos, y_pos = np.clip([x_pos, y_pos], 0.0, 1.0 - 1e-4)  # subtract small value to avoid index out of range
                        row = int(y_pos * output_rows)
                        col = int(x_pos * output_cols)
                        y[row][col][0] = confidence
                        y[row][col][1] = (x_pos - float(col) / output_cols) / (1.0 / output_cols)
                        y[row][col][2] = (y_pos - float(row) / output_rows) / (1.0 / output_rows)
                        y[row][col][i+3] = 1.0
                batch_y.append(y)
            batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
            batch_y = np.asarray(batch_y).reshape((self.batch_size,) + self.output_shape).astype('float32')
            yield batch_x, batch_y

    def get_next_image_path(self):
        path = self.image_paths[self.image_index]
        self.image_index += 1
        if self.image_index == len(self.image_paths):
            np.random.shuffle(self.image_paths)
            self.image_index = 0
        return path

    @staticmethod
    def load_img(path, color=True):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        if color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        return img, path

    @staticmethod
    def resize(img, size):
        if size[0] > img.shape[1] or size[1] > img.shape[0]:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

