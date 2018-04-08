---
layout: page
title: Онлайн курс "Программирование глубоких нейронных сетей на Python"
permalink: /courses/nnpython
---
Глубокие нейронные сети в настоящее время являются одним из самых популярных методов интеллектуального анализа данных. Почти во всех предметных областях они показывают более качественные результаты, по сравнению с другими методами машинного обучения. 

Отличительная особенность курса заключается в том, что он ориентирован на практическое использование нейронных сетей, а не изучение их внутреннего устройства. Для понимания курса не требуется глубокое знание математики. Курс рассчитан на программистов, способных успешно применять существующие библиотеки глубоких нейронных сетей для решения практических задач анализа изображений и текстов.

**Обновление от 10.11.2017**. *Бэкенд Keras заменен на TensorFlow в связи с объявлением об прекращении разработки Theano. Все примеры кода теперь используют бэкенд TensorFlow. Рекомендуется [заменить Theano на TensorFlow](/deep_learning/2017/11/11/Deep-Learning-Course-TensorFlow.html), а для новой установки [сразу использовать TensorFlow](/deep_learning/2017/09/07/Keras-Installation-TensorFlow.html)*.

## Структура курса

Курс состоит из видеолекций и практических работ. 

В **лекциях** изложены теоретические основы работы глубоких нейронных сетей и особенности их обучения, описаны популярные в настоящее время типы глубоких нейронных сетей (сверточные сети, сети долго-краткосрочной памяти (LSTM)), библиотеки для языка Python, реализующие глубокие нейронные сети ([Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [Theano](http://deeplearning.net/software/theano/)), а также методы использования глубоких нейронных сетей для анализа изображений и текстов.

**Практические работы** содержат задания для самостоятельного выполнения на анализ открытых наборов данных ([MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) и [IMDB Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)) с использованием глубоких нейронных сетей и примеры программ, которые можно использовать в качества образца.

## Материалы курса

### Основы обучения глубоких нейронных сетей

1. [Введение](/courses/nnpython-intro).
2. [Лекция "Искусственные нейронные сети"](https://youtu.be/lACoEv1qe1U).
3. [Лекция "Обучение нейронных сетей"](https://youtu.be/KunK-QcqgOg).
4. [Лекция "Библиотеки для глубокого обучения"](https://youtu.be/9xfPb2hiqNY).
5. [Лекция "Распознавание рукописных цифр"](https://youtu.be/0ImpTjNeWGo).
6. [Лекция "Анализ качества обучения нейронной сети"](https://youtu.be/ykDH66b0N_4).
7. [Практическая работа "Распознование рукописных цифр из набора данных MNIST на Keras"](/courses/nnpython-lab1) ([ноутбук в Colaboratory](https://drive.google.com/file/d/11OmSvPZvkEiALlLhMJRo0kgHUv1mpepf/view?usp=sharing)).
7. [Лекция "Решение задачи регрессии"](https://youtu.be/hgvnvWCoDYo) ([ноутбук в Colaboratory](https://drive.google.com/file/d/1F5EoQmzHJ9GIxIFHi6AxBhBwpTvbJWa0/view?usp=sharing)).
8. [Лекция "Сохранение обученной нейронной сети"](/deep_learning/2017/02/12/How-to-save-trained-deep-net.html).
9. [Лекция "Сверточные нейронные сети"](https://youtu.be/52U4BG0ENiM).
10. [Лекция "Распознавание объектов на изображениях"](https://youtu.be/5GdtghjJ3-U).
11. [Практическая работа "Распознавание объектов на изображениях с помощью Keras"](/courses/nnpython-lab2) ([ноутбук в Colaboratory](https://drive.google.com/file/d/1nA3KIasI3DT4E9DsMiiPDoqFFiwapsme/view?usp=sharing)).
12. [Лекция "Рекуррентные нейронные сети"](https://youtu.be/38iGggnbbsQ).
13. [Лекция "Анализ текстов с помощью рекуррентных нейронных сетей"](https://youtu.be/7Tx_cewjhGQ). 
14. [Практическая работа "Определение тональности отзывов на фильмы с помощью Keras"](/courses/nnpython-lab3).
15. [Лекция "Использование GPU для ускорения обучения нейронной сети"](/deep_learning/2017/03/11/How-to-use-gpu-with-theano.html).

### Глубокие нейронные сети для задач компьютерного зрения

1. [Лекция "Анализируем изображения с помощью нейронных сетей"](/deep_learning/2017/06/20/Image-Classification-Using-Neural-Networks.html).
2. [Лекция "Предварительно обученные нейронные сети в Keras"](/deep_learning/2017/06/06/Keras-Pretrained-Networks.html).
3. [Лекция "Как подготовить собственный набор изображений для обучения нейронной сети в Keras"](/deep_learning/2018/01/06/How-to-Prepare-Image-Dataset-for-Keras.html).
4. [Лекция "Перенос обучения (Transfer Learning)"](/deep_learning/2018/01/08/Transfer-Learning-in-Keras.html).
5. [Практика "Практика: Распознавание собак и кошек на изображениях"](/courses/nnpython-lab4).
6. [Лекция "Тонкая настройка нейронной сети"](/deep_learning/2018/04/02/Fine-Tuning-in-Keras.html).
7. [Проект "Распознавание человека по лицу на фотографии"](/deep_learning/2017/08/11/Foto-Verification-with-Dlib.html).

### Анализ текстов с помощью глубоких нейронных сетей

Новый раздел, скоро будут подготовлены материалы.

## Необходимое программное обеспечение

Используется библиотека [Keras](https://keras.io/), а также [TensorFlow](https://www.tensorflow.org/) или [Theano](http://deeplearning.net/software/theano/) в качестве вычислительного бэкенда. Все библиотеки распространяются бесплатно. 

Примеры кода протестированы на Python 3 и TensorFlow. Инструкции по установке Keras:

- Установка Keras и TensorFlows с дистрибутивом Anaconda. [Вариант 1](/deep_learning/2017/09/07/Keras-Installation-TensorFlow.html), [вариант 2](/deep_learning/2018/03/30/TensorFlow-Anaconda-Pip-Install.html).
- [Использование Keras в TensorFlow 1.4](/deep_learning/2017/12/14/How-to-Use-Keras-in-TensorFlow-14.html).

Также для запуска примеров кода можно использовать [платформу Google Colaboratory](/deep_learning/2018/04/04/Google-Colaboratory-for-Deep-Learning.html), где все необходимые библиотеки уже установлены. Google предоставляет Colaboratory бесплатно.

## Практические примеры использования глубоких нейронных сетей

1. [Сверточная нейронная сеть для распознавания рукописных цифр MNIST](/deep_learning/2017/05/08/CNN-for-MNIST.html).
2. [Соревнования по распознаванию рукописных цифр MNIST на Kaggle](/deep_learning/2017/05/10/MNIST-On-Kaggle.html).

## Примеры программ

[https://github.com/sozykin/dlpython_course](https://github.com/sozykin/dlpython_course).

## Дополнительные материалы

1. [Математика глубоких нейронных сетей](/deep_learning/2017/08/31/Math-of-Deep-Learning.html). Список книг и статей для тех, кто хочет разобраться с математическими основами глубоких нейронных сетей.

## Благодарности

При реализации проекта используются средства поддержки, выделенные в качестве гранта на основании конкурса, проведенного Общероссийской общественно-государственной просветительской организации «Российское общество «Знание».
