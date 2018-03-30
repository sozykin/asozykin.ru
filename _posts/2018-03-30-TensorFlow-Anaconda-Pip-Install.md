---
layout: post
title:  "Установка TensorFlow в Anaconda с помощью pip"
date:   2018-03-30 09:00:00 +0500
categories: deep_learning
comments: true
---
В Anaconda, самом популярном дистрибутиве Python для машинного обучения, анализа данных и научных вычислений, пакеты устанавливаются с помощью `conda`. К сожалению, данный подход не всегда работает при установке TensorFlow. Если у вас не получилось [установить TensorFlow стандартными средствами Anaconda](/deep_learning/2017/09/07/Keras-Installation-TensorFlow.html), можно попробовать использовать `pip`. Именно такой метод установки официально [рекомендуется на сайте TensorFLow](https://www.tensorflow.org/install/).

<!--more-->

## Установка Anaconda

Скачайте и установите [Anaconda](https://www.anaconda.com/download/). Выберите вариант для своей операционной системы, python версии 3. Обратите внимание, что нужно устанавливать 64-битную версию python. С 32-битной версией TensorFlow не работает.

TensorFlow можно установить в двух вариантах: для CPU и для GPU. Нужно выбрать один из этих двух вариантов. Обучать сколько-нибудь серьезные нейронные сети без GPU за разумный срок невозможно. Поэтому если у вас есть GPU компании NVIDIA, то выбирайте установку для GPU. В противном случае вам подойдет установка для CPU.

## Установка для CPU

1. **Установка TensorFlow для CPU**:

        pip install --ignore-installed --upgrade tensorflow

2. (*не обязательно*) **Установка Keras**. Начиная с версии 1.4 [TensorFlow включает Keras](/deep_learning/2017/12/14/How-to-Use-Keras-in-TensorFlow-14.html). Поэтому отдельно устанавливать Keras не обязательно. Но если есть желание, то это можно сделать:

        pip install keras

3. **Проверка установки**. Для проверки корректности установки TensorFlow, запустите python и выполните программу:

        import tensorflow as tf
        hello = tf.constant('Hello, TensorFlow!')
        sess = tf.Session()
        print(sess.run(hello))

    В результате должно быть напечатано:

        b'Hello, TensorFlow!'
    
## Установка для GPU

1. **Проверка GPU**. Перед установкой убедитесь, что у вас есть GPU, поддерживаемый TensorFlow. Нужен GPU компании NVIDIA с CUDA Compute Capability 3.5 и выше. Узнать Compute Capability своего GPU можно на [сайте NVIDIA](https://developer.nvidia.com/cuda-gpus). К сожалению, видеокарты AMD и других производителей для TensorFlow не подходят.

2. **Установка CUDA**. Сейчас TensorFlow поддерживает CUDA 9.0, инструкция по установке находится в [отдельной статье](/deep_learning/2018/02/06/How-to-Install-Cuda-9-on-Windows-10.html). Обратите внимание, что нужно устанавливать точные версии, которые поддерживает TensorFlow, а не последние доступные. С версией CUDA 9.1 в настоящее время TensorFlow не работает. Также для TensorFlow обязательно установить библиотеку cuDNN.

3. **Установка TensorFlow для GPU**:

        pip install --upgrade tensorflow-gpu

4. (*не обязательно*) **Установка Keras**:

        pip install keras

5. **Проверка установки**. Для проверки корректности установки TensorFlow, запустите python и выполните программу:

        import tensorflow as tf
        tf.test.gpu_device_name()
    
    На моем ноутбуке с видеокартой NVIDIA Geforce GTX 1050i выдается следующая информация: 

        2018-03-30 10:21:20.514979: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1344] Found device 0 with properties:
        name: GeForce GTX 1050 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.62
        pciBusID: 0000:01:00.0
        totalMemory: 4.00GiB freeMemory: 3.29GiB
        2018-03-30 10:21:20.520164: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1423] Adding visible gpu devices: 0
        2018-03-30 10:22:21.202845: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:911] Device interconnect StreamExecutor with strength 1 edge matrix:
        2018-03-30 10:22:21.206380: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:917]      0
        2018-03-30 10:22:21.208050: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:930] 0:   N
        2018-03-30 10:22:21.211735: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1041] Created TensorFlow device (/device:GPU:0 with 3028 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
        '/device:GPU:0'


    Видно, что TensorFlow нашел видеокарту и создал для нее устройство `device:GPU:0`.

## Полезные ссылки

1. [Дистрибутив Python от Anaconda](https://www.anaconda.com/download/).
2. [Установка TensorFlow и Keras в Anaconda](/deep_learning/2017/09/07/Keras-Installation-TensorFlow.html).
3. [Инструкция по установке на сайте TensorFlow](https://www.tensorflow.org/install/).
4. [Учебный курс "Программирование глубоких нейронных сетей на Python"](/courses/nnpython).
