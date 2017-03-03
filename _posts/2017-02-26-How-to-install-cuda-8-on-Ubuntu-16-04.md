---
layout: post
title:  "Как установить CUDA 8 на Ubuntu 16.04"
date:   2017-02-26 09:00:00 +0500
categories: deep_learning
comments: true
---
Обучение нейронной сети можно значительно ускорить используя GPU. Theano и Tensorflow могут обучать нейронные сети на GPU компании NVIDIA. Для этого нужно установить [NVIDIA CUDA](http://www.nvidia.ru/object/cuda-parallel-computing-ru.html). В этой статье я расскажу о том, как установить CUDA 8 на Ubuntu 16.04. Установку CUDA для Windows опишу позже в отдельной статье.

<!--more-->

# Удаление предыдущих версий CUDA и драйверов NVIDIA

Если у вас на компьютере была установлена предыдущая версию CUDA или драйверы NVIDIA, то их необходимо удалить.

При установке из пакетов:

    sudo apt-get remove --purge nvidia-*

При установке из запускаемого файла (runfile) от NVIDIA:

    sudo nvidia-uninstall
    
# Установка драйвера NVIDIA

Устанавливать драйвер NVIDIA GPU удобнее всего из репозитория [Proprietary GPU Drivers](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa). 

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    sudo apt-get install nvidia-375

# Установка CUDA 8

В репозитории Ubuntu сейчас есть только CUDA 7.5. Так как мы хотим использовать CUDA 8, то придется устанавливать вручную. Загрузите запускаемый файл (runfile) CUDA 8 с [сайта NVIDIA](https://developer.nvidia.com/cuda-downloads). 

Для установки запустите файл. Обратите внимание, что не нужно устанавливать драйверы GPU, которые входят в состав этого файла.

    sudo sh ./cuda_8.0.61_375.26_linux.run

CUDA 8 будет установлена в каталог `/usr/local/cuda-8.0`. Дополнительно будет создана символическая ссылка `/usr/local/cuda`. Рекомендуется использовать именно эту ссылку, чтобы в будущем можно было быстро изменить используемую версию CUDA.

# Настройка параметров окружения

После установки необходимо прописать пути к исполняемым файлам и библиотекам CUDA. Для этого создаем файл `/etc/profile.d/cuda.sh` и записываем туда следующее:
    
    export PATH=$PATH:/usr/local/cuda/bin
    export CUDADIR=/usr/local/cuda
    export GLPATH=/usr/lib

Для библиотек создаем файл `/etc/ld.so.conf.d/cuda.conf` с одной строкой:

    /usr/local/cuda/lib64
    
Затем для настройки библиотек нужно вызвать команду: 

    sudo ldconfig

# Перезагрузка

На этом установка завершена. Необходимо перезагрузить компьютер, чтобы можно было использовать CUDA.

# Итоги

Мы рассмотрели, как устанавливать CUDA 8 на Ubuntu 16.06. В следующих статьях я расскажу о том, как использовать CUDA для ускорения обучения нейронных сетей в Theano и Tensorflow.

# Ссылки

1. [NVIDIA CUDA (Compute Unified Device Architecture)](http://www.nvidia.ru/object/cuda-parallel-computing-ru.html).
2. [Загрузка CUDA 8](https://developer.nvidia.com/cuda-downloads).
3. [Install Ubuntu 16.04 or 14.04 and CUDA 8 and 7.5 for NVIDIA Pascal GPU](https://www.pugetsystems.com/labs/hpc/Install-Ubuntu-16-04-or-14-04-and-CUDA-8-and-7-5-for-NVIDIA-Pascal-GPU-825/).



