---
layout: post
title:  "Установка Keras в Anaconda"
date:   2016-12-25 16:00:00 +0500
categories: deep_learning
comments: true
---
**Обновление от 10.11.2017**: *Все примеры курса ["Программирование глубоких нейронных сетей на Python"](http://www.asozykin.ru/courses/nnpython) переведены на TensorFlow в качестве бэкенда для Keras. Инструкция по [установке Keras с TensorFlow](/deep_learning/2017/09/07/Keras-Installation-TensorFlow.html)*.

Для выполнения практических заданий по курсу ["Программирование глубоких нейронных сетей на Python"](http://www.asozykin.ru/courses/nnpython) необходимо установить библиотеку [Keras](https://keras.io/), а также один из вычислительных бекендов для этой библиотеки - [Theano](http://deeplearning.net/software/theano/) или [TensorFlow](https://www.tensorflow.org/). Самый простой способ это сделать - установить диструбутив Python Anaconda и после этого установить все необходимые пакеты с помощью conda. 

<!--more-->

**Обновление от 17.04.2017**: *изменено на Keras версии 2, Theano 0.9 и Python 3.6*.

1. **Установка Anaconda**. Сначала необходимо установить диструбутив Python Anaconda. Скачайте с [сайта Continuum Analytics](https://www.continuum.io/downloads) версию Anaconda для своей операционной системы. Выбирайте версию Python 3.6.

2. **Установка Theano**. Все примеры в курсе протестированы с библиотекой Theano. Чтобы установить Theano в Anaconda, выполните команду:
    
        conda install theano
  
3. **Установка Keras**. В Linux выполните команду:
  
        conda install keras

    Для Windows пока нет Keras в основном наборе пакетов Anaconda, но можно установить из conda-forge:

        conda install -c conda-forge keras

4. **Настраиваем Keras на работу с Theano**. В файле `.keras/keras.json`, который находится в домашнем каталоге пользователя, прописываем Theano в качестве бекенда:

        {
            "epsilon": 1e-07,
            "backend": "theano",
            "image_data_format": "channels_first",
            "floatx": "float32"
        }
  

5. **Проверка установки**. Напечатаем версию Keras, которая была установлена:

        python -c "import keras; print(keras.__version__)"
    
    Результат должен быть примерно таким:
    
        Using Theano backend.
        2.0.2
    
    Установлена версия Keras 2.0.2, в качестве бекенда используется Theano.
    
На этом установка закончена, можно запускать [примеры кода из курса](https://github.com/sozykin/dlpython_course).
 
Если у вас не получается установить Keras, пишите свои вопросы в комментариях. Постараюсь помочь.
     

