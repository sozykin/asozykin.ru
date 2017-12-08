---
layout: post
title:  "Как установить библиотеку OpenCV в Python"
date:   2017-12-08 10:00:00 +0500
categories: computer_vision
comments: true
---
[OpenCV](https://opencv.org/) - это популярная библиотека компьютерного зрения. Библиотека написана на C++, но имеет API для Python. Преимущества Python в том, что на нем можно быстро сделать прототип приложения компьютерного зрения. Кроме того, школьникам, на которых будет ориентирована серия моих статей о компьютерном зрении, с Python работать значительно проще, чем с C++.

[Официальная инструкция по установке OpenCV в Python](https://docs.opencv.org/3.3.1/d5/de5/tutorial_py_setup_in_windows.html) довольно запутанная и описывает установку только для Python 2.7. Если вы используете третий Python, то она вам не подойдет. К счастью, есть простой способ установить OpenCV с помощью Anaconda как для второй, так и для третьей версии Python.

<!--more-->

## Установка OpenCV

С использованием Anaconda OpenCV для Python можно установить всего за два простых шага.

1. **Установка Anaconda**. Скачайте с сайта [Anaconda](https://www.anaconda.com/download/) дистрибутив для нужной вам операционной системы и версии Python. Установите Anaconda на свой компьютер.

2. **Установка OpenCV**. Установить OpenCV можно с использованием менеджера пакетов `conda`. Для этого выполните следующую команду:

        conda install -c conda-forge opencv

    `conda` установит из канала conda-forge библиотеку OpenCV и все необходимые зависимости. 


## Проверка работоспособности OpenCV

Чтобы проверить, успешно ли установилась OpenCV, запустим небольшую программу, которая захватывает видео с Web-камеры и показывает его на экране:

```python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

cap.release()
cv2.destroyAllWindows()
```
       
Я пробовал устанавливать Anaconda 5.0.1 с Python 3.6 в Windows 10 и Ununtu 16.04, в обоих случаях все заработало.

## Полезные ссылки

1. [Библиотека OpenCV](https://opencv.org/).
2. [Дистрибутив Python от Anaconda](https://www.anaconda.com/download).
3. [Учебник по OpenCV на Python](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html).
