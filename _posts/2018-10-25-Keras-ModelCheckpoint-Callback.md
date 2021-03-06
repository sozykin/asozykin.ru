---
layout: post
title:  "Сохранение нейросети в процессе обучения"
date:   2018-10-25 11:00:00 +0500
categories: deep_learning
comments: true
---
{% include youtube-player.html id="vLnBMM-DAYY" %}

Обучение нейронной сети, как правило, требует значительного времени, поэтому важно сохранять обученную сеть для дальнейшего использования. Но иногда бывает что веса, полученные на последней эпохе обучения сети, не являются лучшими. Например, у нас началось переобучение и обобщающая способность сети стала снижаться. Можно перезапустить процесс обучения с меньшим количеством эпох, но это не является хорошим решением если обучение идет долго. Альтернативный вариант -- использовать [`ModelCheckpoint Callback`](https://keras.io/callbacks/#modelcheckpoint), который позволяет сохранять веса нейронной сети на каждой эпохе обучения.

<!--more-->

## Демонстрационная нейросеть для распознавания рукописных цифр

Давайте рассмотрим, как применить `ModelCheckpoint Callback` на примере нейронной сети для [распознавания рукописных цифр из набора данных MNIST](/courses/nnpython-lab1).

На первом этапе нужно подключить интересующий нас callback совместно с другими модулями Keras:

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
```

Загружаем данные и создаем нейронную сеть:

```python
# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразуем в нужный формат
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = utils.to_categorical(y_train, 10)
Y_test = utils.to_categorical(y_test, 10)

# Создаем последовательную модель для нейронной сети
model = Sequential()

# Полносвязная нейронная сеть, состоящая из двух слоев
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Компилируем модель
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", 
              metrics=["accuracy"])
```

## Создаем ModelCheckpoint сallback

ModelCheckpoint callback создается следующим образом:

```python
сheckpoint = ModelCheckpoint('save/mnist-dense-{epoch:02d}-{val_acc:.4f}.hdf5')
```

При создании callback нужно указать путь, куда будут сохраняться веса моделей. Путь задается с помощью шаблона, который передается в виде параметра при создании `ModelCheckpoint callback`. В начале шаблона находится префикс, одинаковый для всех эпох: `save/mnist-dense-`. Модели будут записываться в каталог `save`, имя файла начинается с `mnist-dense-` -- полносвязная сеть для распознавания рукописных цифр из набора данных MNIST. 

На каждой эпохе обучения нужно сохранять сеть в отдельный файл, поэтому вторая часть шаблона содержит переменные. `{epoch:02d}` будет заменена на номер эпохи (целое число с двумя знаками), а `{val_acc:.4f}` -- на долю верных ответов на проверочном наборе данных (число с плавающей точкой, 4 знака после запятой). Вместо `val_acc` можно указывать другие метрики, если вы их используете, как на обучающем, так и на проверочном наборах данных, а также значение функции ошибки.


## Запускаем обучение с сохранением сети на каждой эпохе

Чтобы сеть сохранялась на каждой эпохе обучения, при вызове метода `fit` в параметре `callbacks` мы указываем созданный ранее `сheckpoint`. Не забудьте перед запуском создать каталог `save`, куда будут записываться модели, иначе обучение остановится на первой эпохе из-за ошибки записи в несуществующий каталог.

```python
history = model.fit(X_train, 
                    Y_train, 
                    batch_size=200, 
                    epochs=25, 
                    validation_split=0.2, 
                    verbose=2, 
                    callbacks=[сheckpoint])
```

После завершения обучения в каталоге `save` мы получим 25 файлов:

![Список файлов с весами моделей на каждой эпохе](/assets/images/dl_course/saved_models.png)

В имени каждого файла есть номер эпохи и доля верных ответов на проверочном наборе данных. В моем случае самый высокий показатель 0.9830 был на 22 эпохе. После этого доля верных ответов начала снижаться, что говорит о переобучении.

## Сохранение только лучшей сети

Полносвязная сеть для распознавания рукописных цифр MNIST занимает мало места, ее можно сохранять на каждой эпохе. Но что делать, если вы обучаете крупную сеть с большим количеством весов, сохранять которую на каждом этапе не эффективно? Для этого случая `ModelCheckpoint callback` предоставляет возможность сохранения только одного состояния нейронной сети с лучшей метрикой. Такой режим работы включается, если указать параметр `save_best_only=True`:

```python
сheckpoint = ModelCheckpoint('save/mnist-dense.hdf5', 
                              monitor='val_acc', 
                              save_best_only=True)
```

Будет сохраняться только лучшее состояние сети, так что вместо шаблона указывается имя файла -- `save/mnist-dense.hdf5`. Параметр `monitor` показывает, какая метрика будет использоваться для определения лучшего состояния. В примере это `val_acc` -- доля верных ответов на проверочном множестве. Как и в предыдущем случае, можно использовать любую метрику, которую вы применяете, а также значение ошибки.

## Итоги

`ModelCheckpoint сallback` в Keras позволяет сохранить нейронную сеть в процессе обучения. Это полезно, если обучение сети занимает длительное время. Если вы указали слишком много эпох и началось переобучение, то вам не придется перезапускать обучение сети заново. 

Есть два режима работы `ModelCheckpoint callback`: 
- сохранение сети на каждой эпохе обучения.
- сохранение одного файла с лучшим вариантом сети на основе заданной метрики.

## Полезные ссылки

1. [Полный текст примера кода использования ModelCheckpoint Callback для распознавания рукописных цифр](https://github.com/sozykin/dlpython_course/blob/master/keras_callbacks/saving_model.ipynb).
2. [Документация на `ModelCheckpoint Callback`](https://keras.io/callbacks/#modelcheckpoint).
