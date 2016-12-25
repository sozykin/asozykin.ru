---
layout: post
title:  "Установка Keras в Anaconda"
date:   2016-12-25 16:00:00 +0500
categories: deep_learning
comments: true
---
Для выполнения практических заданий по курсу ["Программирование глубоких нейронных сетей на Python"](http://www.asozykin.ru/courses/nnpython) необходимо установить библиотеку [Keras](https://keras.io/), а также один из вычислительных бекендов для этой библиотеки - [Theano](http://deeplearning.net/software/theano/) или [TensorFlow](https://www.tensorflow.org/). Самый простой способ это сделать - установить диструбутив Python Anaconda и после этого установить все необходимые пакеты с помощью conda. 

<!--more-->

1. **Установка Anaconda**. Сначала необходимо установить диструбутив Python Anaconda. Скачайте с [сайта Continuum Analytics](https://www.continuum.io/downloads) версию Anaconda для своей операционной системы. Выбирайте третью версию Python, потому что именно она используется в курсе.

2. **Установка Theano**. Все примеры в курсе протестированы с библиотекой Theano. Чтобы установить Theano в Anaconda, выполните команду:  
    
    `conda install theano`
  
3. **Установка Keras**. Пока Keras не входит в основной набор пакетов Anaconda, но его можно установить из conda-forge:
  
    `conda install -c conda-forge keras`

4. **Настраиваем Keras на работу с Theano**. В файле `.keras/keras.json`, который находится в домашнем каталоге пользователя, прописываем Theano в качестве бекенда:

    ```
    {
        "epsilon": 1e-07,
        "backend": "theano",
        "image_dim_ordering": "th",
        "floatx": "float32"
    }
    ```
    
    Также указываем, что будем использовать порядок хранения измерений в изображениях, который применяется в Theano (`"image_dim_ordering": "th"`). В TensorFlow использиуется другой порядок.

5. **Проверка установки**. Напечатаем версию Keras, которая была установлена:

    `python -c "import keras; print(keras.__version__)"`
    
    Результат должен быть примерно таким:
    
    ```
    Using Theano backend.
    1.0.7
    ```
    
    Установлена версия Keras 1.0.7, в качестве бекенда используется Theano.
    
 На этом установка закончена, можно запускать примеры кода из курса.
 
 Если у вас не получается установить Keras, пишите свои вопросы в комментариях. Постараюсь помочь.
     

