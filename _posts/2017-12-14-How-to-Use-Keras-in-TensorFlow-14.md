---
layout: post
title:  "Как использовать Keras в TensorFlow 1.4"
date:   2017-12-14 09:00:00 +0500
categories: deep_learning
comments: true
---
Keras [входит в состав TensorFlow начиная с версии 1.2](https://blog.keras.io/introducing-keras-2.html), а в версии TensorFlow 1.4 его [перевели из contrib в core packages](https://developers.googleblog.com/2017/11/announcing-tensorflow-r14.html). Это означает, что Keras готов к продуктивному использованию в составе TensorFlow. Отдельно устанавливать Keras не нужно.

Чтобы использовать Keras из TensorFlow, нужно поменять импорт модулей. Вместо модуля `keras` используйте `tensorflow.python.keras`. Например, импорт для [распознавания рукописных цифр набора данных MNIST](/courses/nnpython-lab1) будет выглядеть так:

```python
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils import np_utils
```

Больше ничего в программе менять не нужно.

## Полезные ссылки

1. [Google Developers Blog: Announcing TensorFlow r1.4](https://developers.googleblog.com/2017/11/announcing-tensorflow-r14.html).
2. [How to import keras from tf.keras in Tensorflow?](https://stackoverflow.com/questions/47262955/how-to-import-keras-from-tf-keras-in-tensorflow).

