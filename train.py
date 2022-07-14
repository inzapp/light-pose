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
from human_pose_estimator import HumanPoseEstimator

if __name__ == '__main__':
    HumanPoseEstimator(
        train_image_path=r'/train_data/pose/train',
        validation_image_path=r'/train_data/pose/validation',
        input_shape=(96, 96, 1),
        lr=0.002,
        momentum=0.9,
        batch_size=32,
        iterations=300000).fit()
