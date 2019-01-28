---
layout: post
title:  "Соревнования по распознаванию рукописных цифр MNIST на Kaggle"
date:   2017-05-10 22:00:00 +0500
categories: deep_learning
comments: true
---
{% include youtube-player.html id="zO0RAtZRkpc" %}

**Обновление от 27.01.2019**. *Добавлено видео, расширение данных, использование функций callback, ноутбук в Colab. Архитектура сети заменена на более эффективную*.

Хотите проверить, как хорошо вы научились распознавать рукописные цифры из набора данных MNIST? Попробуйте свои силы в соревнованиях на сайте Kaggle!

[Kaggle.com](https://kaggle.com) - это сайт для Data Scientist'ов. На нем регулярно проводятся различные соревнования по машинному обучению, есть большое количество открытых наборов данных для анализа, а также форум, на котором общаются Data Scientist'ы. Одно из соревнований связано с [распознаванием рукописных цифр MNIST](https://www.kaggle.com/c/digit-recognizer). Вы можете загрузить свое решение на сайт соревнования и посмотреть, насколько хорошо работает ваша модель по сравнению с моделями других участников. Давайте попробуем применить глубокие нейронные сети, которые мы изучали в [курсе по программированию нейронных сетей на Python](/courses/nnpython), для соревнования по MNIST на Kaggle. Как обычно, мы будем использовать библиотеку Keras. Полный текст программы есть в [репозитории курса на github](https://github.com/sozykin/dlpython_course/blob/master/mnist/kaggle_mnist.ipynb), а также на облачной платформе [Google Colaboratory](https://colab.research.google.com/drive/1qovAFIaTaMJzJroCwx20bK1pTBb6e8Uy).

<!--more-->

## Подключаем необходимые библиотеки

На первом этапе подключаем необходимые библиотеки: Keras из TensorFlow, sklearn, numpy и matplotlib:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten 
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
```


## Исходные данные

Формат исходных данных в соревнованиях Kaggle отличается от [стандартного формата MNIST Яна Лекуна](http://yann.lecun.com/exdb/mnist/). На Kaggle изображения MNIST представлены в виде обычных текстовых файлов, которые можно скачать на закладке [Data](https://www.kaggle.com/c/digit-recognizer/data) страницы соревнования. Данные включают два файла:

- train.csv - данные для обучения.
- test.csv - данные для предсказания.

Файл train.csv содержит описание изображений MNIST в текстовом формате по одному изображению в каждой строке файла. Формат файла следующий: в первой позиции цифра, которая представлена на изображении, а затем через запятую 784 кода интенсивности пикселей изображения в оттенках серого (размер изображения 28х28). Вот пример одной строки файла:
  
    6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,181,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,166,254,177,...

На первом месте правильный ответ - цифра 6. Затем через запятую коды пикселов (для сокращения показаны не все 784). Таких строк в файле 42 тысячи.

Файл test.csv имеет такой же формат, но в нем нет правильного ответа в первой позиции. Наша программа должна сама определить, какая цифра на изображении. Так как правильные ответы заранее не известны, то проверить качество обучения сети самостоятельно мы не можем. Для этого необходимо загрузить файл с нашими предсказаниями цифр для каждого изображения на [страницу соревнования на Kaggle](https://www.kaggle.com/c/digit-recognizer/submit), где правильность распознавания проверится автоматически. За сутки можно загрузить не более 5 вариантов решений. Загружать нужно текстовый csv-файл в следующем формате:

    ImageId,Label
    1,2
    2,0
    3,9
    4,9
    5,3

В первом столбце указывается номер изображения, а во втором - цифра на нем, которую мы распознали.

В файле test.csv 28 тыс. изображений. Всего в двух файлах 70 тыс. изображений, как в оригинальном наборе данных MNIST.

## Загрузка данных для обучения

Загрузить данные в формате csv с помощью встроенных средств работы с набором данных MNIST в Keras нельзя. Поэтому будем читать данные из csv-файла с помощью библиотеки numpy, функции loadtxt:

    train_dataset = np.loadtxt('train.csv', skiprows=1, delimiter=',')

С помощью этой команды мы загружаем текстовый файл, в котором в качестве разделителя используется запятая (`delimiter=','`) и пропускаем первую строку, которая содержит заголовок с описанием столбцов файла (`skiprows=1`). В результате этой команды создается массив `numpy`, в котором 785 столбцов (первый столбец - цифра на картинке, затем 784 пиксела изображения) и 42 тысячи строк.

Из набора данных выделяем данные о картинках и меняем формат из плоского вектора 784 пикселя на двумерную матрицу 28х28:

```python
# Выделяем данные для обучения
x_train = train_dataset[:, 1:]
# Переформатируем данные в 2D, бэкенд TensorFLow
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Нормализуем данные
x_train /= 255.0
```

Выделяем правильные ответы (метки классов) и преобразуем их в представление по категориям:

```python
# Выделяем правильные ответы
y_train = train_dataset[:, 0]
# Преобразуем правильные ответы в категоризированное представление
y_train = utils.to_categorical(y_train)
```

Разделяем набор данных на две части: для обучения (`X_Train`, `Y_train`) и проверки (`X_val`, `Y_val`):
```python
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(x_train, 
                                                  y_train, 
                                                  test_size = 0.1, 
                                                  random_state=random_seed)
```

# Дополнение данных

После того, как мы разделили набор данных на две части, у нас осталось не очень много данных для обучения -- всего 37800 изображений. Для качественного обучения этого недостаточно. Поэтому, чтобы не возникло переобучение, мы будем использовать [дополнение данных](https://youtu.be/mCHoMsner54). В Keras это делается с помощью генераторов:

```python
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)
```

Генератор `datagen` работает с изображениями (`ImageDataGenerator`) и позволяет получать новые картинки из имеющихся путем следующих преобразований:
- Поворот на случайный угол до 10 градусов (`rotation_range=10`).
- Увеличение размера до 10 процентов (`zoom_range = 0.10`).
- Сдвиг по ширине и высоте на случайное значение до 10 процентов (`width_shift_range=0.1` и `height_shift_range=0.1`).

Генератор создает изображения с использованием нескольких вариантов преобразований. Например, изображение может быть повернуто влево на 5 градусов и увеличено на 7 процентов, или сдвинуто вверх на 5 процентов и повернуто вправо на 8 градусов. Вот несколько примеров сгенерированных изображений:

![Пример дополнения данных из набора MNIST](/assets/dl/mnist_gen.png)

# Создание и обучение нейронной сети

Для распознавания рукописных цифр для Kaggle мы будем использовать сверточную нейронную сеть из Kaggle Kernel [Introduction to CNN Keras - Acc 0.997 (top 8%)](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6). Она задается следующим образом:

```python
# Создаем последовательную модель
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Компилируем модель
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", 
              metrics=["accuracy"])
```

У нас задача классификации, поэтому в качестве функции активации на последнем слое сети используем [softmax](/deep_learning/2018/10/26/Softmax-Function.html), а в качеству функции ошибки -- категориальную перекрестную энтропию.

В процессе обучения мы будем использовать две callback функции: `ModelCheckpoint` и `ReduceLROnPlateau`. Первая функция используется для [сохранения нейросети в процессе обучения](/deep_learning/2018/10/25/Keras-ModelCheckpoint-Callback.html) и задается следующим образом:

```python
сheckpoint = ModelCheckpoint('mnist-cnn.hdf5', 
                              monitor='val_acc', 
                              save_best_only=True,
                              verbose=1)
```

Нейросеть сохраняется в файл с именем `mnist-cnn.hdf5` только если текущее значение доли верных ответов на проверочном множестве (`monitor='val_acc'`) лучше, чем на предыдущих эпохах (задается параметром `save_best_only=True`). Параметр `verbose=1` говорит о том, что функция печатает лог своих действий.

Функция `ReduceLROnPlateau` используется для уменьшения скорости обучения, если в процессе обучения не происходит улучшения качество работы сети (выход на плато).

```python
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```

Мы наблюдаем за долей верных ответов на проверочном наборе данных (`monitor='val_acc'`) и если в течение трех эпох этот показатель не увеличился (`patience=3`), то параметр скорости обучения умножается на 0.5 (`factor=0.5`). Так продолжается, пока параметр скорости обучения не достигнет 0.00001 (`min_lr=0.00001`).

Для обучения нейронной сети используется генератор `datagen`, который мы задали в предыдущем разделе:

```python
batch_size=96
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size), 
                    epochs=30,
                    validation_data=(X_val, Y_val),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    verbose=1,
                    callbacks=[сheckpoint, learning_rate_reduction])
```

Обратите внимание, что для проверки не используется дополнение данных.

## Загрузка лучшего варианта нейронной сети

Обучение продолжается в течение 30 эпох, что немного больше, чем нужно. Поэтому на последних эпохах возникает переобучение. Обнаружить переобучение можно по журналу, выдаваемому в процессе обучения:

```
Epoch 29/30
391/393 [============================>.] - ETA: 0s - loss: 0.0171 - acc: 0.9942
Epoch 00029: val_acc improved from 0.99595 to 0.99619, saving model to mnist-cnn.hdf5
393/393 [==============================] - 14s 37ms/step - loss: 0.0171 - acc: 0.9942 - val_loss: 0.0141 - val_acc: 0.9962
Epoch 30/30
391/393 [============================>.] - ETA: 0s - loss: 0.0158 - acc: 0.9949
Epoch 00030: val_acc did not improve from 0.99619
393/393 [==============================] - 14s 36ms/step - loss: 0.0157 - acc: 0.9949 - val_loss: 0.0144 - val_acc: 0.9955
```

На 29 эпохе доля верных ответов на проверочном множестве улучшилась с 0.99595 до 0.99619, поэтому сеть была сохранена. Но на 30 эпохе этот показатель снизился до 0.9955. Таким образом, после 30 эпох обучения сеть обеспечивает не лучшее качество работы. Поэтому перед распознаванием рукописных цифр из тестового набора нам нужно загрузить лучший вариант сети:

```python
model.load_weights('mnist-cnn.hdf5')
```

Теперь модель готова для распознавания.

## Распознавание рукописных цифр

Изображения из файла train.csv, цифры на которых нам нужно распознать, загрузим также с помощью функции `loadtxt` библиотеки `numpy`:

```python
# Загружаем данные для предсказания
test_dataset = np.loadtxt('test.csv', skiprows=1, delimiter=",")
# Переформатируем данные в 2D, бэкенд TensorFlow
x_test = test_dataset.reshape(test_dataset.shape[0], 28, 28, 1)
x_test /= 255.0
```

Теперь, когда мы обучили сеть и загрузили данные для тестирования, можно выполнять распознавание. Для этого используем метод `model.predict`:

```python
# Распознаем цифры на изображениях
predictions = model.predict(x_test)
# Преобразуем предсказания из категориального представления в метки классов
predictions = np.argmax(predictions, axis=1)
```

Результаты распознавания необходимо записать их в текстовый файл, который мы будем загружать на Kaggle для проверки правильности работы:

```python
out = np.column_stack((range(1, predictions.shape[0]+1), predictions))
np.savetxt('submission.csv', out, header="ImageId,Label", 
            comments="", fmt="%d,%d")
```

На первом шаге мы используем функцию `column_stack` библиотеки `numpy` для подготовки данных в нужном формате: номер изображения и цифра на нем. Затем записываем полученные результаты в файл c помощью функции `savetxt`.

Полученный файл можно загрузить на страницу ["Submit Predictions"](https://www.kaggle.com/c/digit-recognizer/submit) соревнований Kaggle и проверить, насколько хорошо ваша программа распознает рукописные цифры по сравнению с другими пользователями.

## Расскажите о своих результатах

Расскажите в комментариях, какое место в соревновании по распознаванию MNIST на Kaggle вам удалось занять? Какие изменения вы внесли в программу, чтобы этого достичь? 

## Полезные ссылки

1. [Соревнования по распознаванию рукописных цифр MNIST на Kaggle](https://www.kaggle.com/c/digit-recognizer).
2. [Ноутбук с полным кодом решения на облачной платформе Google Colaboratory](https://colab.research.google.com/drive/1qovAFIaTaMJzJroCwx20bK1pTBb6e8Uy).
5. [Репозиторий с примерами программ из статьи](https://github.com/sozykin/dlpython_course/blob/master/mnist/kaggle_mnist.ipynb).
2. [Учебный курс "Программирование глубоких нейронных сетей на Python"](/courses/nnpython).
3. [Инструкция по использованию платформы Google Colaboratory](/deep_learning/2018/04/04/Google-Colaboratory-for-Deep-Learning.html).
3. [Дополнение данных для нейросети](https://youtu.be/mCHoMsner54).
4. [Сохранение нейронной сети в процессе обучения](/deep_learning/2018/10/25/Keras-ModelCheckpoint-Callback.html).


