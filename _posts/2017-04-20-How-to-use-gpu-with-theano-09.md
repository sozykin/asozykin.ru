---
layout: post
title:  "Как использовать GPU в Theano 0.9 и Keras 2"
date:   2017-04-20 09:00:00 +0500
categories: deep_learning
comments: true
---
В новой версии Theano 0.9 был изменен бэкенд GPU. Теперь для этой цели Theano использует `gpuarray`. Настройки GPU в Theano поменялись, однако саму программу на Keras менять для подключения GPU по-прежнему не нужно.

Конфигурационный файл `.theanorc` для использования GPU в Theano 0.9:

```
[global]
floatX = float32
device = cuda

[gpuarray]
preallocate = 0.8
```

<!--more-->

Изменилось название устройства: вместо `gpu` теперь нужно писать `cuda`. Также вместо библиотеки CNMeM теперь используются возможности `gpuarray` для предварительного выделения памяти. Параметр `gpuarray.preallocate` задает долю памяти GPU, которая будет выделена для обучения нейронной сети (в примере 80% памяти GPU). 

В сообщениях Theano должна появиться информация об использовании GPU с бэкендом `cuda`:

```
Using Theano backend.
Using cuDNN version 5110 on context None
Preallocating 4856/6070 Mb (0.800000) on cuda
Mapped name None to device cuda: GeForce GTX 1060 (0000:01:00.0)
```

Формат диагностических сообщений поменялся по сравнению с предыдущей версией бэкенда. Сначала создается контекст cuda, а затем он отображается на реальную карту GeForce GTX 1060. Память выделяется с помощью preallocate - 80%, что составляет 4856 из 6070 мегабайт встроенной памяти видеокарты. 

Новый бэкенд также использует библиотеку [cuDNN](/deep_learning/2017/03/20/cuDNN-with-theano-and-keras.html) для ускорения обучения нейронных сетей на GPU. Причем Theano 0.9 поддерживает новую версию библиотеки cuDNN 5.1. Предыдущая версия Theano работала только с cuDNN 5.0.

Основное преимущество нового бэкенда - возможность работы с несколькими GPU. К сожалению, пока я не успел этого попробовать. Но обязательно сделаю в ближайшее время и напишу, что получилось. 

# Полезные ссылки

1. [Theano Tutorial. Using the GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html).
2. [Установка CUDA в Ubuntu 16.04](/deep_learning/2017/02/26/How-to-install-cuda-8-on-Ubuntu-16-04.html).
3. [Установка CUDA в Windows 10](/deep_learning/2017/03/08/How-to-install-cuda-8-on-Windows-10.html).
4. [Как использовать GPU в Theano и Keras](/deep_learning/2017/03/11/How-to-use-gpu-with-theano.html).
5. [Библиотека cuDNN для быстрого обучения нейронных сетей на GPU](/deep_learning/2017/03/20/cuDNN-with-theano-and-keras.html).



