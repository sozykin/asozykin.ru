---
layout: post
title:  "Библиотека cuDNN для быстрого обучения нейронных сетей на GPU"
date:   2017-03-20 10:00:00 +0500
categories: deep_learning
comments: true
---
Мы рассмотрели, как [можно увеличить скорость обучения нейронной сети с помощью CUDA и GPU](/deep_learning/2017/03/11/How-to-use-gpu-with-theano.html). Для сети, которая [распознает объекты из CIFAR-10](/courses/nnpython-lab2), применение GPU позволило ускорить обучение [более чем в 7 раз, по сравнению с CPU](/deep_learning/2017/03/11/How-to-use-gpu-with-theano.html). Однако это не предел, и обучение можно сделать еще быстрее, если использовать библиотеку [NVIDIA cuDNN](https://developer.nvidia.com/cudnn).

<!--more-->

# Что такое cuDNN

В отличии от [NVIDIA CUDA](http://www.nvidia.ru/object/cuda-parallel-computing-ru.html), которая обеспечвает возможность выполнения на GPU любых вычислений, библиотека cuDNN специально разработана для обучения глубоких нейронных сетей. Она содержит оптимизированные для GPU реализации сверточных и рекуррентных сетей, различных функций активации (полулинейная, сигмоидальная, гиперболический тангенс, softmax), алгоритма обратного распространения ошибки и т.п. cuDNN позволяет обучать нейронные сети на GPU в несколько раз быстрее, чем просто CUDA.   

# Установка cuDNN

Перед установкой cuDNN необходимо убедиться, что у вас есть GPU компании NVIDIA, и установить CUDA. Инструкции по установке CUDA для [Ubuntu 16.04](/deep_learning/2017/02/26/How-to-install-cuda-8-on-Ubuntu-16-04.html) и для [Windows 10](/deep_learning//2017/03/08/How-to-install-cuda-8-on-Windows-10.html).

Библиотека cuDNN распространяется бесплатно. Чтобы ее получить, нужно зарегистрироваться на [сайте разработчиков NVIDIA](https://developer.nvidia.com).

Перед загрузкой не забудьте уточнить, какую версию cuDNN поддерживает ваша система глубокого обучения. Это можно сделать на [странице со списком поддерживаемых cuDNN систем](https://developer.nvidia.com/deep-learning-frameworks). Во время написания этой статьи, TensorFlow поддерживает cuDNN версии 5.1, а Theano только версии 5. Устанавливать более новую версию не имеет смысла, т.к. Theano отказывается с ней работать.

Скачать cuDNN можно по [ссылке](https://developer.nvidia.com/cudnn). Выберите необходимую вам версию и операционную систему. Для Theano и Keras я скачал версию 5 для Linux и для Windows. Оба варианта заработали.

Скачанный архив с cuDNN нужно распаковать и скопировать его содержимое в каталог с установленной CUDA. Для Linux это чаще всего `/usr/local/cuda`, а для Windows - `c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`.   

# Тестируем производительность cuDNN

Для теста производительности cuDNN будем обучать нейронную сеть, которая [распознает на изображениях объекты из набора данных CIFAR-10](/courses/nnpython-lab2). В [предыдущей статье](/deep_learning/2017/03/11/How-to-use-gpu-with-theano.html) были получены следующие результаты:

- Обучение на CPU Intel Core i7-6700HQ (2.60ГГц, 4 ядра) - 152 секунды на эпоху.
- Обучение на GPU NVIDIA GeForce GTX 1060 c CUDA и CNMeM - 20 секунд на эпоху.

Попробуем запустить обучение этой сети на NVIDIA GeForce GTX 1060 с cuDNN:

```
$ python cifar10.py 
Using Theano backend.
Using gpu device 0: GeForce GTX 1060 (CNMeM is enabled with initial size: 80.0% of memory, cuDNN 5005)
Train on 45000 samples, validate on 5000 samples
Epoch 1/25
7s - loss: 1.8344 - acc: 0.3172 - val_loss: 1.4191 - val_acc: 0.4812
Epoch 2/25
7s - loss: 1.3934 - acc: 0.4919 - val_loss: 1.1873 - val_acc: 0.5782
Epoch 3/25
7s - loss: 1.2305 - acc: 0.5597 - val_loss: 1.0567 - val_acc: 0.6242
```

В диагностическом выводе мы видим, что используется GPU с CNMeM и cuDNN:
   
    Using gpu device 0: GeForce GTX 1060 (CNMeM is enabled with initial size: 80.0% of memory, cuDNN 5005)

Время обучения одной эпохи - 7 секунд. Это почти в 3 раза быстрее, чем на GPU без cuDNN, и в 21 раз быстрее, чем на CPU.

# Итоги

Библиотека cuDNN содержит оптимизированные для GPU функции обучения нейронных сетей. Она позволяет обучать нейронную сеть в несколько раз быстрее, чем с использованем CUDA, и в несколько десятков раз быстрее, чем на CPU. Скачать cuDNN можно бесплатно, для этого необходимо предварительно [зарегистрирваться как разработчик на сайте NVIDIA](https://developer.nvidia.com).

# Полезные ссылки

1. [NVIDIA cuDNN](https://developer.nvidia.com/cudnn).
2. [NVIDIA CUDA](http://www.nvidia.ru/object/cuda-parallel-computing-ru.html).
3. [Сайт разработчиков NVIDIA](https://developer.nvidia.com).

