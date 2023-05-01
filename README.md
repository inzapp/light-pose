# Human Pose Estimator

Human Pose Estimator is a top-down key point detection model that inspired by the yolo object detection model

This is a one-stage model in which both feature extraction and key point detection are performed through a simple convolution network

Unlike other keypoint detection models, this model is characterized by high coordinate accuracy because it additionally learns offset for grid

It also learns low confidence for human-invisible-keypoints, resulting in significantly less FP

For these reasons, models tend to learn more about the shape of a person

For example, if the input image is an upper body image,

The model predicts only key points for the upper body and not unnecessary key points for the lower body

The more diverse the training data, the more robust the model is

<img src="/md/sample.jpg" width="500"><br>

## Augmentation

In addition to basic augmentation, rotation augmentation can be used to improve the performance of the model

The model is the same image, but by further learning the rotated image, overfitting is avoided and more generalized

<img src="/md/augmentation.jpg" width="800"><br>

We also provide scripts that can perform this augmentation simply

## Loss function

This model uses ALE loss, an improved version of Binary Crossentropy loss

See [**absolute-logarithmic-error**](https://github.com/inzapp/absolute-logarithmic-error)

## Labeling

What labeling tools should I use to make training data?

This model provides a dedicated labeling tool, label_pose

<img src="/md/label_pose.gif" width="500"><br>

Here's how to use it

d : next image<br>
a : previous image<br>
e : next limb point<br>
q : previous limb point<br>
w : toggle to show skeleton line<br>
f : auto find and go to not labeled image<br>
x : remove current label content<br>
left click : set limb point<br>
right click : remove limb point<br>
ESC : exit program
