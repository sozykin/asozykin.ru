---
layout: post
title:  "Как установить CUDA 8 на Windows 10"
date:   2017-03-08 21:30:00 +0500
categories: deep_learning
comments: true
---
Вторая статья по установке CUDA для ускорения обучения нейронных сетей на GPU с помощью Theano и Tensorflow. В предыдущей статье я рассказывал, как [установить CUDA на Ubuntu 16.04](/deep_learning/2017/02/26/How-to-install-cuda-8-on-Ubuntu-16-04.html), сейчас мы рассмотрим Windows.

<!--more-->

## Установка Microsoft Visual Studio

CUDA содержит компилятор nvcc, который может генерировать код для GPU, но не для CPU. Поэтому для работы CUDA нужен компилятор для CPU. В Linux для этой цели используется gcc, а в Windows - Microsoft Visual Studio. К сожалению, под Windows CUDA не может использовать gcc, даже если вы его [установили вместе с Theano](/deep_learning/2016/12/25/Keras-Installation.html). Поэтому необходимо обязательно установть Microsoft Visual Studio.

CUDA 8 [поддерживает Visual Studio](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows) версий 2015, 2013 и 2012. Причем для 2015 версии поддерживается бесплатный вариант Visual Studio Community. Я установил именно его. Скачать можно с сайта [my.visualstudio.com](https://my.visualstudio.com), нужна предварительная регистрация. 

Обратите внимание, что необходимо устанавливать Visual Studio на **английском языке**! Theano не может работать с русским Visual Studio (и вообще с любой версией, которая выдает сообщения в кодировке, отличной от английской). 

При установке Visual Studio выберите средства для разработки C++. Остальное по желанию.

После установки добавьте путь к компилятору Visual Studio (cl.exe) в переменную PATH. 

## Установка CUDA 8

CUDA 8 нужно скачать с [сайта NVIDIA](https://developer.nvidia.com/cuda-downloads). Для Windows CUDA распространяется в виде запускаемого exe файла, в котором есть драйвер для GPU с поддержкой CUDA и сама CUDA 8. Просто запустите скачанный файл и установите все по-умолчанию.

## Перезагрузка

После установки Microsoft Visual Studio и CUDA 8 компьютер необходимо перезагрузить.

## Проверка установки

Для проверки работоспособности CUDA запустите утилиту nvidia-smi (находится в каталоге C:\Program Files\NVIDIA Corporation\NVSMI). Вывод должен выглядеть примерно так:

```
C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi.exe
Wed Mar 08 21:27:54 2017
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 376.51                 Driver Version: 376.51                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1060   WDDM  | 0000:01:00.0      On |                  N/A |
| N/A   49C    P3    20W /  N/A |    165MiB /  6144MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      1032  C+G   Insufficient Permissions                     N/A      |
|    0      3776  C+G   ... Files (x86)\ASUS\Giftbox\Asusgiftbox.exe N/A      |
|    0      5420  C+G   ...ost_cw5n1h2txyewy\ShellExperienceHost.exe N/A      |
|    0      5436  C+G   ...am Files (x86)\Dropbox\Client\Dropbox.exe N/A      |
|    0      5868  C+G   C:\Windows\explorer.exe                      N/A      |
|    0     11108  C+G   ...indows.Cortana_cw5n1h2txyewy\SearchUI.exe N/A      |
+-----------------------------------------------------------------------------+
```

## Полезные ссылки

1. [NVIDIA CUDA (Compute Unified Device Architecture)](http://www.nvidia.ru/object/cuda-parallel-computing-ru.html).
2. [Загрузка CUDA 8](https://developer.nvidia.com/cuda-downloads).
3. [Загрузка Microsoft Visual Studio](https://my.visualstudio.com).



