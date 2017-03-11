---
layout: post
title:  "Как использовать GPU в Theano и Keras"
date:   2017-03-11 21:30:00 +0500
categories: deep_learning
comments: true
---

Сократить время обучения нейронной сети в несколько раз можно за счет использования GPU (Graphics Processing Unit). Современные GPU подходят не только для обработки графики и компьютерных игр, но и для вычислений. GPU содержит сотни (а некоторые модели тысячи) вычислительных ядер, поэтому расчеты на нем выполняются гораздо быстрее, чем на CPU. 

Многие современные системы обучения нейронных сетей поддерживают GPU. Среди таких систем Theano и Tensorflow, поэтому Keras тоже работает с GPU. Приятная особенность в том, что вам не придется переделывать программу на Keras, требуется лишь изменить настройки вычислительного бекенда.  

В этой статье я расскажу как использовать GPU в Theano. Я буду рассматривать текущую стабильную версию Theano 0.8. В следующей версии 0.9 появился [новый бекенд для GPU и настройки изменились](http://deeplearning.net/software/theano_versions/dev/tutorial/using_gpu.html).

<!--more-->

# Проверяем, что у вас есть подходящий GPU

Theano использует [технологию CUDA](http://www.nvidia.ru/object/cuda-parallel-computing-ru.html), которая работает только на GPU производства компании NVIDIA. Если у вас GPU другого производителя, то использовать его в Theano, к сожалению, не получится.   

# Измеряем производительность на CPU

Перед началом использования GPU измерим производительность на CPU. Для тестов будем использовать [сверточную сеть распознавания объектов на изображениях из набора данных CIFAR-10](/courses/nnpython-lab2). Сеть обучается достаточно долго, поэтому использование GPU может дать существенный прирост производительности. На моем процессоре Intel Core i7-6700HQ (2.60ГГц, 4 ядра) одна эпоха обучения занимает от 152 до 154 секунд:

```
Using Theano backend.
Train on 45000 samples, validate on 5000 samples
Epoch 1/25
152s - loss: 1.8658 - acc: 0.3048 - val_loss: 1.4747 - val_acc: 0.4626
Epoch 2/25
153s - loss: 1.3849 - acc: 0.4968 - val_loss: 1.2194 - val_acc: 0.5712
Epoch 3/25
154s - loss: 1.2142 - acc: 0.5645 - val_loss: 1.0771 - val_acc: 0.6250
```

# Устанавливаем CUDA

Чтобы можно было использовать GPU в Theano, необходимо установить CUDA. Инструкции по установке:

- [Ubuntu 16.04](/deep_learning/2017/02/26/How-to-install-cuda-8-on-Ubuntu-16-04.html).
- [Windows 10](/deep_learning/2017/03/08/How-to-install-cuda-8-on-Windows-10.html).

# Конфигурируем Theano

Есть несколько способов [настроить использование GPU в Theano](http://deeplearning.net/software/theano/tutorial/using_gpu.html). Самый быстрый - добавить флаги Theano в строку запуска:

    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py

Здесь мы указываем Theano использовать GPU (флаг device=gpu) и в качестве типа данных float использовать float32 (флаг floatX=float32), т.к.  float64 на GPU Theano не поддерживает. 

Чтобы не прописывать флаги при каждом запуске, можно использовать конфигурационный файл Theano. Он называется `.theanorc` и должен находиться в домашнем каталоге пользователя (под Windows допускается использовать файл `.theanorc.txt`). В этот файл нужно включить следующие строки:

```
[global]
floatX = float32
device = gpu
```

Параметры те же самые, что и во флагах в предыдущем примере. 

Запускаем наш тестовый пример с CIFAR-10:

```
Using Theano backend.
Using gpu device 0: GeForce GTX 1060 (CNMeM is disabled, cuDNN not available)
Train on 45000 samples, validate on 5000 samples
Epoch 1/25
27s - loss: 1.8421 - acc: 0.3184 - val_loss: 1.3848 - val_acc: 0.4922
Epoch 2/25
27s - loss: 1.3733 - acc: 0.5005 - val_loss: 1.2078 - val_acc: 0.5774
Epoch 3/25
27s - loss: 1.1989 - acc: 0.5723 - val_loss: 1.0369 - val_acc: 0.6362
```

Библиотека Theano сообщила нам, что используется устройство GPU с номером 0: NVIDIA GeForce GTX 1060. Время обучения одной эпохи сократилось до 27 секунд, примерно в 5,5 раз меньше, чем на GPU. Хороший результат, но его можно еще улучшить.

# Настраиваем CNMeM

В предыдущем выводе Theano мы можем обратить внимание на сообщение `(CNMeM is disabled, cuDNN not available)`. [CNMeM](https://github.com/NVIDIA/cnmem) - это менеджер памяти, разработанный компанией NVIDIA, чтобы помочь системам глубокого обучения управлять памятью GPU. CNMeM позволяет повысить производительность обучения нейронных сетей на GPU.    

Для использования CNMeM в Theano не нужно ничего дополнительно устанавливать. Включить CNMeM можно в конфигурационном файле .theanorc:

```
[global]
floatX = float32
device = gpu

[lib]
cnmem = 0.8
```

Возможные значения параметра cnmem:

- 0 - CNMeM отключена.
- больше 0, но меньше 1 - CNMeM включена, число указывает долю памяти GPU, которая будет использована для вычислений.
- 1 - CNMeM включена, вся память GPU используется для вычислений.

Если у вас один GPU для вычислений и для графики, то для Theano рекомендуется выделять не более 80% памяти GPU (`cnmem = 0.8`). Если же у вас есть отдельный GPU для расчетов, то можно установить `cnmem = 1`. 

Запускаем тест с CIFAR-10 (я использую значение `cnmem = 0.8`):

```
Using Theano backend.
Using gpu device 0: GeForce GTX 1060 (CNMeM is enabled with initial size: 80.0% of memory, cuDNN not available)
Train on 45000 samples, validate on 5000 samples
Epoch 1/25
20s - loss: 1.8421 - acc: 0.3184 - val_loss: 1.3848 - val_acc: 0.4922
Epoch 2/25
20s - loss: 1.3733 - acc: 0.5005 - val_loss: 1.2078 - val_acc: 0.5774
Epoch 3/25
20s - loss: 1.1989 - acc: 0.5723 - val_loss: 1.0369 - val_acc: 0.6362
```

Видно, что библиотека CNMeM используется (`CNMeM is enabled with initial size: 80.0% of memory`). Время обучения одной эпохи сократилось до 20 секунд, что в 7,6 раза быстрее, чем обучение на CPU.

# Итоги

Применение GPU позволило нам ускорить обучение нейронной сети в 7,6 раза. Для этого потребовалось установить CUDA и включить CNMeM.

Производительность обучения можно еще увеличить, если использовать [библиотеку cuDNN](https://developer.nvidia.com/cudnn). Что это за библиотека, как ее установить и использовать, я расскажу в одной из следующих статей. 


# Полезные ссылки

1. [Theano Tutorial. Using the GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html).
2. [Установка CUDA в Ubuntu 16.04](/deep_learning/2017/02/26/How-to-install-cuda-8-on-Ubuntu-16-04.html).
3. [Установка CUDA в Windows 10](/deep_learning/2017/03/08/How-to-install-cuda-8-on-Windows-10.html).
4. [CNMeM](https://github.com/NVIDIA/cnmem).
4. [Тестовый пример. Распознавание объектов на изображениях из набора CIFAR-10](/courses/nnpython-lab2).


