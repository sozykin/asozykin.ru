---
layout: post
title:  "Установка Keras с TensorFlow в Anaconda"
date:   2017-09-07 08:00:00 +0500
categories: deep_learning
comments: true
---
![Keras TensorFlow Anaconda](/assets/images/anaconda_keras_tf.png)

Библиотека [TensorFlow](https://www.tensorflow.org/) от Google стремительно развивается и завоевывает популярность. Более того, [Google решил включить Keras](http://www.fast.ai/2017/01/03/keras/) в TensorFlow. Поэтому я адаптирую все примеры курса ["Программирование глубоких нейронных сетей на Python"](http://www.asozykin.ru/courses/nnpython) для TensorFlow. Но сначала нужно установить эти библиотеки и настроить их на совместную работу. Как и в случае с [Theano](/deep_learning/2016/12/25/Keras-Installation.html), самый простой и удобный способ сделать это -- использовать диструбутив Python Anaconda. 

<!--more-->

## Установка Anaconda

Сначала необходимо установить Anaconda. Скачайте с [сайта Continuum Analytics](https://www.continuum.io/downloads) и установите версию Anaconda для своей операционной системы. Выбирайте Python версии 3.6.

## Установка TensorFlow для CPU

Для установки TensorFlow используйте команду:

    conda install tensorflow

TensorFlow для CPU установится автоматически со всеми необходимыми зависимостями. Проверяем, что установка прошла успешно:

    python -c "import tensorflow; print(tensorflow.__version__)"
    1.2.1

## Установка Keras 

После установки TensorFlow можно устанавливать Keras. Для Linux используйте команду:

    conda install keras

В Windows Keras пока не входит в основной набор пакетов Anaconda, поэтому устанавливаем его из conda-forge:

    conda install -c conda-forge keras

По-умолчанию после установки Keras будет сконфигурирован на работу с TensorFlow. На всякий случай проверим содержимое конфигурационного файла `.keras/keras.json` в домашнем каталоге пользователя:

    {
        "floatx": "float32",
        "epsilon": 1e-07,
        "image_data_format": "channels_last",
        "backend": "tensorflow"
    }

В `backend` должно быть `tensorflow`, а в `image_data_format` -- значение `channels_last` (формат хранения изображений в TensorFlow).

Проверяем, что все работает:

    python -c "import keras; print(keras.__version__)"
    Using TensorFlow backend.
    2.0.5

Установлена версия Keras 2.0.5, используется бекенд TensorFlow. Можно переходить к [обучению глубоких нейросетей](/courses/nnpython).

## Установка Keras и TensorFlow для GPU

Если у вас есть GPU, то обучение нейронных сетей с его помощью можно [ускорить в несколько десятков раз](/deep_learning/2017/03/20/cuDNN-with-theano-and-keras.html). К сожалению, подойдет только GPU производства компании NVIDIA и только с Compute Capability 3.0 и выше. Узнать Compute Capability своей карты можно на [сайте NVIDIA](https://developer.nvidia.com/cuda-gpus). Если у вас старая видеокарта, то в качестве бекенда для Keras можно [использовать Theano](/deep_learning/2017/03/11/How-to-use-gpu-with-theano.html), у которой меньше требования по Compute Capability. 

1. **Устанавливаем драйвер GPU**. Первое, что нужно сделать -- это установить драйвер для GPU. При этом CUDA и cuDNN устанавливать не обязательно, они установятся автоматически из пакетов `conda`. Для Windows в дополнение к драйверу нужно установить Visual Studio. Можно воспользоваться инструкциями для [Linux](/deep_learning/2017/02/26/How-to-install-cuda-8-on-Ubuntu-16-04.html) или [Windows](/deep_learning/2017/03/08/How-to-install-cuda-8-on-Windows-10.html), но не устанавливайте CUDA.

2. **Установка Keras** выполняется также, как и для CPU. Для Linux:

        conda install keras

    Для Windows:

        conda install -c conda-forge keras

    Перед продолжением установки рекомендуется проверить работоспособность Keras на CPU, как было описано выше.

3. **Установка TensorFlow для GPU**. Установка выполняется одной командой, зависимости cuda-toolkit, cuDNN и некоторые другие устанавливаются автоматически с помощью менеджера пакетов `conda`:

        conda install tensorflow-gpu

    После установки Keras будет автоматически использовать версию TensorFlow для GPU. Ничего дополнительно настраивать не нужно.

4. **Проверка доступности GPU в TensorFlow**. TensorFlow выдает диагностические сообщения о том, какое устройство используется для расчетов. В отличие от Theano, эти сообщения не видны в Jupyter Notebook, их нужно искать в консоли сервера Jupyter. Примерный вид диагностических сообщений: 
        
        2017-11-10 21:01:19.946649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties: 
        name: GeForce GTX 1060
        major: 6 minor: 1 memoryClockRate (GHz) 1.6705
        pciBusID 0000:01:00.0
        Total memory: 5.93GiB
        Free memory: 5.41GiB
        2017-11-10 21:01:19.946680: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0 
        2017-11-10 21:01:19.946685: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y 
        2017-11-10 21:01:19.946692: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
    
    TensorFlow использует устройство номер 0 с названием GeForce GTX 1060. Выводится также информация и тактовой частоте и объеме доступной памяти GPU.

    На этом установка для GPU завершена, можно запускать [примеры кода из курса](/courses/nnpython). Примеры работают как на CPU, так и на GPU, менять программы не нужно.

Если у вас не получается установить Keras и TensorFlow, пишите свои вопросы в комментариях. Постараюсь помочь.

## Дополнительные ссылки

1. Курс ["Программирование глубоких нейронных сетей на Python"](http://www.asozykin.ru/courses/nnpython).
2. Инструкция по установке [Keras с Theano](/deep_learning/2016/12/25/Keras-Installation.html).
3. Установка [CUDA 8 в Windows 10](/deep_learning/2017/03/08/How-to-install-cuda-8-on-Windows-10.html).
4. Установка [CUDA 8 в Ubuntu 16.04](/deep_learning/2017/02/26/How-to-install-cuda-8-on-Ubuntu-16-04.html).
5. [Использование GPU в Theano и Keras](/deep_learning/2017/03/11/How-to-use-gpu-with-theano.html).
6. [Библиотека cuDNN](/deep_learning/2017/03/20/cuDNN-with-theano-and-keras.html).
     

