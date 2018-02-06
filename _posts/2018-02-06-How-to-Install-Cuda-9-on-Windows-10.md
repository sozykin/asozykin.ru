---
layout: post
title:  "Установка CUDA 9 в Windows 10 для TensorFlow и Keras"
date:   2018-02-06 11:00:00 +0500
categories: deep_learning
comments: true
---
Недавно вышла версия [TensorFlow 1.5](https://developers.googleblog.com/2018/01/announcing-tensorflow-15.html) с поддержкой CUDA 9, поэтому можно переводить TensorFlow и Keras на новую версию CUDA. В этой статье я расскажу, как установить CUDA 9 и CuDNN 7 в Windows 10. По установке для Linux будет отдельная статья.

## Что нужно устанавливать

Чтобы TensorFlow и Keras могли использовать GPU под Windows, необходимо установить три компонента:

1. **Microsoft Visual Studio**. Любая GPU-программа содержит код как для GPU, так и для CPU. Данные для проведения расчетов нужно загрузить из файлов и передать их в память GPU, где они будут обработаны. Результаты обработки передаются обратно на CPU для сохранения и визуализации. NVIDIA СUDA включает компилятор только для GPU. Код для CPU генерируется с помощью внешнего компилятора, для Windows это Microsoft Visual Studio.
2. **NVIDIA CUDA** - библиотека, которая позволяет использовать GPU для проведения вычислений общего назначения (general purpose computing), а не только обрабатывать графику.
3. **Библиотека cuDNN**. Это библиотека для CUDA, которая содержит эффективные реализации операций с нейронными сетями. В отличие от Theano, TensorFlow не может работать без cuDNN. 

<!--more-->

## Проверка GPU

CUDA работает только с GPU компании NVIDIA, вот [список поддерживаемых GPU](https://developer.nvidia.com/cuda-gpus). Если у вас GPU от AMD или Intel, то CUDA работать не будет и TensorFlow, к сожалению, не сможет использовать такие GPU.

Для TensorFlow нужна видеокарта с CUDA Compute Capability 3.0. Узнать Compute Capability для своей карты можно на [сайте](https://developer.nvidia.com/cuda-gpus). Если у вас старый GPU с Compute Capability меньше 3.0, то вместо TensorFlow можно использовать Theano, у которой меньше требования к GPU. Однако ускорение на таких GPU будет не очень большим, по сравнению с центральным процессором.

## Установка Microsoft Visual Studio

Под Windows NVIDIA CUDA использует Microsoft Visual Studio для генерации кода для CPU. Список поддерживаемых версий Visual Studio для CUDA 9 приведен на сайте [NVIDIA](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). На момент написания этой статьи поддерживаются версии:

- Visual Studio 2017
- Visual Studio 2015
- Visual Studio Community 2015
- Visual Studio 2013
- Visual Studio 2012

Из бесплатных версий есть только Visual Studio Community 2015, я устанавливал именно ее. Скачать Visual Studio Community 2015 можно с сайта [my.visualstudio.com](https://my.visualstudio.com/), нужна предварительная регистрация. 

Если у вас есть Visual Studio 2017, то можно использовать ее. Однако обратите внимание, что пока не поддерживается обновленная версия Visual Studio 2017 с компилятором Visual C++ 15.5. Официально CUDA 9 работает только с Visual C++ 15.0. 

Рекомендую устанавливать Visual Studio на английском языке. С русскоязычной Visual Studio есть проблемы у Theano. В TensorFlow я пока с подобными проблемами не сталкивался, но, возможно, мне просто повезло.

При установке Visual Studio выберите средства для разработки C++, остальное устанавливайте по желанию.

После установки добавьте путь к компилятору Visual С++ (cl.exe) в переменную PATH.

## Установка NVIDIA CUDA

На время написания этой статьи текущей версией CUDA является 9.1. Однако TensorFlow 1.5 поддерживает только CUDA 9.0. Если установите 9.1, то получите ошибку. Странное поведение, но что поделать.

Скачать CUDA 9.0 можно на странице [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). Выбираем нужную версию Windows и тип загружаемого файла - exe(local). Этот файл содержит CUDA и драйвер для GPU с поддержкой CUDA. Просто скачайте файл и запустите его. Все конфигурационные параметры при установке можно оставить по умолчанию.

## Установка cuDNN

cuDNN - это дополнение к NVIDIA CUDA, которое содержит эффективную реализацию операций с нейронным сетями для GPU. DNN в названии библиотеки расшифровывается как Deep Neural Networks. Некоторые библиотеки, например, Theano, могут работать с CUDA без cuDNN. Но TensorFlow так работать не может, для нее cuDNN нужна обязательно.

cuDNN можно скачать бесплатно с [сайта NVIDIA](https://developer.nvidia.com/rdp/form/cudnn-download-survey). Нужна предварительная регистрация в качестве разработчика NVIDIA, что также делается бесплатно и быстро на этом же сайте.

После регистрации будет предложено скачать несколько версий cuDNN для разных версий CUDA. Выбирайте cuDNN 7 для CUDA 9.0. С другими версиями cuDNN и CUDA не будет работать TensorFlow 1.5.

cuDNN представляет собой архив, в котором всего три файла:

- `bin\cudnn64_7.dll`
- `include\cudnn.h`
- `lib\x64\cudnn.lib`

Эти файлы нужно распаковать в каталог, где установлена CUDA 9.0. Как правило, это `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0`.

## Перезагрузка

После установки Microsoft Visual Studio, NVIDIA CUDA и cuDNN, компьютер необходимо перезагрузить.

## Проверка установки CUDA

Проверить, правильно ли установилась CUDA и видит ли она ваш GPU, можно с помощью утилиты `nvidia-smi.exe`. Эта утилита показывает доступные GPU для CUDA. Утилита работает в командной строке и при установке по умолчанию находится в каталоге `c:\Program Files\NVIDIA Corporation\NVSMI`. На моем ноутбуке с GPU NVIDIA GeForce GTX 1050 Ti `nvidia-smi.exe` выдает следующую информацию:

```
Tue Feb 06 13:03:37 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 385.54                 Driver Version: 385.54                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105... WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   35C    P8    N/A /  N/A |     80MiB /  4096MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1224    C+G   ... Files (x86)\Dropbox\Client\Dropbox.exe N/A      |
|    0      6080    C+G   Insufficient Permissions                   N/A      |
|    0      8440    C+G   ...351.0_x64__8wekyb3d8bbwe\Calculator.exe N/A      |
|    0     12664    C+G   C:\Program Files (x86)\SCM\SCM.exe         N/A      |
|    0     14648    C+G   ...4.0_x64__8wekyb3d8bbwe\WinStore.App.exe N/A      |
|    0     15440    C+G   ..._8wekyb3d8bbwe\MessagingApplication.exe N/A      |
|    0     15880    C+G   ...osoft.LockApp_cw5n1h2txyewy\LockApp.exe N/A      |
+-----------------------------------------------------------------------------+
```

В выводе `nvidia-smi.exe` видно, что в компьютере есть один GPU с номером 0, модель GeForce GTX 105... (название видеокарты поместилось не полностью), объем памяти 4 Гб, текущая загрузка 0%. В нижней части показаны процессы, которые используют GPU. 

Видеокарта есть в выводе `nvidia-smi.exe`, это означает, что CUDA установлена успешно.

## Итоги

Кратко, основные особенности установки:

1. Обязательно нужно Microsoft Visual Studio, у меня работает бесплатная англоязычная версия Visual Studio Community 2015. Не забудьте прописать путь с компилятору Visual C++ (cl.exe) в переменную PATH, иначе CUDA его не найдет.
2. Для TensorFlow 1.5 нужно устанавливать CUDA 9.0, а не CUDA 9.1! TensorFlow ищет файлы именно из CUDA 9.0, поэтому CUDA 9.1 не работает.
3. Устанавливать cuDNN обязательно, т.к. без нее TensorFlow не работает. Внимательно смотрим за версиями: нужна cuDNN 7 для CUDA 9.0.


## Полезные ссылки

1. Проверка поддержки [CUDA и Compute Capability](https://developer.nvidia.com/cuda-gpus).
1. Загрузка Visual Studio Community 2015 c [my.visualstudio.com](https://my.visualstudio.com/).
2. Загрузка CUDA 9.0 c [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
3. Загрузка [cuDNN](https://developer.nvidia.com/rdp/form/cudnn-download-survey).


