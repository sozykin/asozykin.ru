---
layout: post
title:  "Анализируем изображения с помощью нейронных сетей"
date:   2017-06-20 22:00:00 +0500
categories: deep_learning
comments: true
---
{% include youtube-player.html id="Z5oMIctZYWk" %}

В курсе "[Глубокое обучение на Python](/courses/nnpython)" мы научились обучать нейронные сети для распознавания рукописных цифр и объектов из набора данных CIFAR-10. Давайте посмотрим, как применять эти нейронные сети для анализа своих изображений.

<!--more-->

# Распознавание рукописных цифр

Начнем с распознавания рукописных цифр. Читатели прислали мне несколько отсканированных картинок с рукописными цифрами. Вот одна из них:

![Рукописная цифра 2](/assets/dl/2.png)

Для распознавания рукописных цифр воспользуемся [сверточной нейронной сетью](/deep_learning/2017/05/08/CNN-for-MNIST.html). 

Keras содержит специальный модуль для работы с изображениями, который называется `image`. С его помощью можно быстро загрузить изображение из файла и преобразовать его в массив `numpy`, который мы можем передать модели для распознавания:

```python
import numpy as np
from keras.preprocessing import image

# Загружаем изображение
img_path = '2.png'
img = image.load_img(img_path, target_size=(28, 28), grayscale=True)

# Преобразуем изображением в массив numpy
x = image.img_to_array(img)

# Инвертируем и нормализуем изображение
x = 255 - x
x /= 255
x = np.expand_dims(x, axis=0)
```

Загружаем из файла обученную [сверточную нейронную сеть](/deep_learning/2017/05/08/CNN-for-MNIST.html):

```python
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("mnist_model.h5")
```

Компилируем модель перед использованием

```python
loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

Теперь мы готовы к распознаванию цифры. Для распознавания вызываем метод модели `predict`:

```python
prediction = loaded_model.predict(x)
```

Модель выдает массив из 10 значений в формате One Hot Encoding. Выбираем индекс максимального значения и печатаем его:

```python
print(np.argmax(prediction))
```

`[2]`

# Распознавание объектов на изображениях

Давайте рассмотрим более сложную задачу - распознавание объектов на изображении. Будем использовать нейронную сеть, обученную на [наборе данных CIFAR-10](/courses/nnpython-lab2). Попробуем распознать картинку самолета, которую сеть не видела в процессе обучения:

![Фотография самолета](/assets/dl/plane.jpg)

Загружаем изображение в Keras:

```python
import numpy as np
from keras.preprocessing import image

img_path = 'plane.jpg'
img = image.load_img(img_path, target_size=(32, 32))
```

В отличие от рукописных цифр, в этот раз изображение цветное и его размер 32х32, в соответствии с форматом CIFAR-10. Преобразуем картинку в массив `numpy`:

```python
x = image.img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)
```

Загружаем сеть, обученную на [наборе данных CIFAR-10](/courses/nnpython-lab2), и компилируем модель:

```python
json_file = open("cifar10_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cifar10_model.h5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Запускаем распознавание объекта:

```python
prediction = loaded_model.predict(x)
```

Для удобства вывода задаем список с названиями классов объектов:

```python
classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
```

Печатаем результат распознавания:

```python
print(classes[np.argmax(prediction)])
```

`[самолет]`

Как видим, нейронная сеть справилась с задачей, несмотря на то, что обучение проводилось на изображениях размером 32х32. Наше изображение пришлось уменьшить до этого размера, но сеть все равно распознала на картинке самолет.


Давайте попробуем дать нейронной сети более сложную задачу - распознать не фотографию объекта, а рисунок. Например, вот такой рисунок лошади:

![Фотография самолета](/assets/dl/horse.png)

Запускаем программу и получаем результат:

`[лошадь]`

Нейронная сеть сумела распознать объект даже на рисунке, хотя мы обучали ее только на фотографиях!

# Итоги

Мы научились применять нейронные сети для анализа любых изображений, а не только тех, которые входят в стандартные наборы данных для обучения. В Keras для этого есть класс `image` из модуля `keras.preprocessing`.

Попробуйте провести эксперименты со своими изображениями и пишите в комментариях, что у вас получилось.

# Полезные ссылки

1. Курс "[Глубокое обучение на Python](/courses/nnpython)".
2. [Сверточная нейронная сеть для распознавания рукописных цифр](/deep_learning/2017/05/08/CNN-for-MNIST.html).
3. [Сверточная нейронная сеть для распознавания объектов из набора данных CIFAR-10](/courses/nnpython-lab2).
4. [Сохранение обученных нейронных сетей](/deep_learning/2017/02/12/How-to-save-trained-deep-net.html).

