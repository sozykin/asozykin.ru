---
layout: page
title: Практика&#58; определение тональности отзывов на фильмы с помощью Keras
permalink: /courses/nnpython-lab3
comments: true
---
В этой практической работе по курсу ["Глубокое обучение на Python"](/courses/nnpython) вы научитесь обучать нейронную сеть определять тональность отзыва на фильмы из базы данных IMDB. 

**Обновление от 08.05.2017**. *Программы обновлены на Keras версии 2*.

**Обновление от 10.11.2017**. *Бэкенд Keras изменен на TensorFlow*.

**Цель работы**: научится оценивать влияние гиперпараметров обучения (количество эпох обучения, количество нейронов на слое LSTM, алгоритм оптимизации) на качество обучения нейронной сети.

## Предварительные сведения

Перед выполнением работы рекомендуется посмотреть видео с объяснением, как работает программа определения тональности отзывов на фильмы из базы данных IMDB.

{% include youtube-player.html id="7Tx_cewjhGQ" %}

## Необходимое программное обеспечение

Используется библиотека [Keras](https://keras.io/), а также [TensorFLow](https://www.tensorflow.org/) в качестве вычислительного бэкенда.

[Инструкция по установке Keras и TensorFlow с дистрибутивом Anaconda](/deep_learning/2017/09/07/Keras-Installation-TensorFlow.html).

Также можно воспользоваться бесплатной облачной платформой для машинного обучения [Google Colaboratory](/deep_learning/2018/04/04/Google-Colaboratory-for-Deep-Learning.html), ссылка на [ноутбук с базовой версией программы](https://drive.google.com/file/d/1ZpA5rZhYBy9HlJbNU2m3Jzl6iN5_Y6cf/view?usp=sharing).

## Базовая версия программы

Базовая версия программы, которая реализует обучение нейронной сети для определения тональности рецензий на фильмы.

```python
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.datasets import imdb

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)
# Максимальное количество слов (по частоте использования)
max_features = 5000
# Максимальная длина рецензии в словах
maxlen = 80

# Загружаем данные
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Заполняем или обрезаем рецензии
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Создаем сеть
model = Sequential()
# Слой для векторного представления слов
model.add(Embedding(max_features, 32))
model.add(SpatialDropout1D(0.2))
# Слой долго-краткосрочной памяти
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
# Полносвязный слой
model.add(Dense(1, activation="sigmoid"))

# Копмилируем модель
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Обучаем модель
model.fit(X_train, y_train, batch_size=64, epochs=7,
          validation_data=(X_test, y_test), verbose=2)
# Проверяем качество обучения на тестовых данных
scores = model.evaluate(X_test, y_test,
                        batch_size=64)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))
```

Также базовую версию программы можно найти в [репозитории примеров курса](https://github.com/sozykin/dlpython_course).

Для начала запустите базовую версию программы:

    python imdb_lstm.py

Примерный вывод программы:

```
Epoch 1/7
25000/25000 [==============================] - 74s - loss: 0.5748 - acc: 0.6947 - val_loss: 0.4217 - val_acc: 0.8117
Epoch 2/7
25000/25000 [==============================] - 75s - loss: 0.4353 - acc: 0.8035 - val_loss: 0.3785 - val_acc: 0.8350
Epoch 3/7
25000/25000 [==============================] - 76s - loss: 0.4001 - acc: 0.8250 - val_loss: 0.3766 - val_acc: 0.8333
Epoch 4/7
25000/25000 [==============================] - 83s - loss: 0.3692 - acc: 0.8396 - val_loss: 0.3615 - val_acc: 0.8390
Epoch 5/7
25000/25000 [==============================] - 84s - loss: 0.3507 - acc: 0.8516 - val_loss: 0.3776 - val_acc: 0.8324
Epoch 6/7
25000/25000 [==============================] - 82s - loss: 0.3380 - acc: 0.8566 - val_loss: 0.3628 - val_acc: 0.8389
Epoch 7/7
25000/25000 [==============================] - 92s - loss: 0.3248 - acc: 0.8619 - val_loss: 0.3664 - val_acc: 0.8413
25000/25000 [==============================] - 24s
Точность на тестовых данных: 84.13%
```

**Запишите точность работы сети после обучения на тестовых данных**. Она приводится в последней строке вывода:

    Точность работы на тестовых данных: 84.13%

**Проанализируйте точность на проверочной выборке в процессе обучения**. Она указывается после заголовка `val_acc`. Началось ли переобучение нейронной сети?

## Экспериментируем с гиперпараметрами обучения

Мы попытаемся улучшить качество обучения сети путем изменения гиперпараметров: количество эпох обучения, количество нейронов на слое LSTM, алгоритм оптимизации. Для этого проведем серию экспериментов, в каждом из которых будем менять один из гиперпараметров, и анализировать, как изменилось качество работы сети.

1. **Количество эпох обучения**. Оценим влияние количества эпох обучения на качество обучения сети. Количество эпох задается в аргументе `epochs` метода `model.fit`:

        model.fit(X_train, y_train, batch_size=64, epochs=7,
          validation_data=(X_test, y_test), verbose=2)
        
    Попробуйте обучать сеть в течение 5, 10 и 15 эпох. Определите, когда начинается переобучение. Выберите количество эпох, при котором самая высокая точность работы сети на тестовых данных.
    
2. **Количество нейронов в слое LSTM**. Оцените влияние количества нейронов в LSTM слое сети на качество обучения. Количество нейронов задается при добавлении слоя в модель:
        
        model.add(LSTM(XXX, dropout=0.2, recurrent_dropout=0.2))
        
    Используйте количество нейронов 50, 125, 150. Выберите количество нейронов в LSTM слое, при котором обеспечивается самая высокая точность обучения. Проанализируйте влияния количества нейронов в LSTM слое на время обучения сети.
    
3. **Алгоритм оптимизации**. Выясним, как влияет алгоритм оптимизации на качество обучения. В базовой версии программы используется эффективный алгоритм оптимизации [Adam](https://arxiv.org/pdf/1412.6980.pdf). Попробуйте заменить его на стандартный алгоритм стохастического градиентного спуска (Stochastic gradient descent, SGD). Для этого поменяйте значение параметра `optimizer` при компиляции модели:

        model.compile(loss='binary_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])
    
## Выбираем лучшие гиперпараметры

Создайте сеть с лучшими значениями всех гиперпараметров обучения, которые вы определили на предыдущем шаге. Увеличилась ли точность работы сети? 

Как сочетание гиперпараметров влияет на время обучения сети? На переобучение?

Что можно сделать, чтобы еще больше увеличить точность?

## Расскажите о своих результатах

Пишите в комментариях, какой точности вам удалось достичь, и какие гиперпараметры вы при этом использовали.
