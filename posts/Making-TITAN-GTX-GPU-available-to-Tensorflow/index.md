---
title: "Making the TITAN GTX GPU available to Tensorflow"
author: "Waseem Waheed"
date: "2020-11-15"
categories: [Python, Linux, Tensorflow, Software Engineering]
---
# Introduction

This post is to document 📝 a process I had to go through. Installing Tensorflow on Ubuntu.

To make use of the GPU(s) in Tensorflow, Linux should be able to detect the GPU and tensorflow-gpu should be installed.

# The process
First of all, you need to check that Linux can detect the GPU as follows:
you can either do:

```console
user# sudo lshw -C display
```
or:
```console
user# nvidia-smi
```
which should show you a list containing all the NVIDIA GPUs attached to your machine.

Next, to check if `tensorflow-gpu` is installed, we can use the following command

```console
user# pip list | grep tensor
tensorboard                       2.4.1
tensorboard-plugin-wit            1.8.0
tensorflow                        1.14.0
tensorflow-estimator              2.3.0
```

as we can see, `tensorflow-gpu` is not in the list. To install it, we use:

```console
user# sudo pip install --upgrade pip
user# pip install tensorflow-gpu --user
```

once finished, we should check the correct operation of Tensorflow as follows:
```console
user# python
```

```python
import tensorflow as tf
print(tf.config.experimental.list_physical_devices('GPU'))
```
which should give you a list of the installed GPUs. To use a specific GPU within a context:

```python
tf.debugging.set_log_device_placement(True)

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:1'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
except RuntimeError as e:
  print(e)
```

To use a specific GPU for all Tensorflow's calculations, use the following template:

```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
```

***Note:*** this note took advantage of Tensorflow's documentation @ https://www.tensorflow.org/guide/gpu

# In the end
I hope that this write up has helped you installing from scratch/fixing your Tensorflow installation.



