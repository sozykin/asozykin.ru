---
layout: post
title:  "Как сохранить обученную нейронную сеть в Keras"
date:   2017-02-12 09:00:00 +0500
categories: deep_learning
comments: true
---
Вы уже наверняка заметили, что обучение нейронной сети - очень долгий процесс, который может занимать несколько часов, дней или даже месяцев. Поэтому желательно иметь возможность использовать сеть для анализа данных, не обучая ее каждый раз. В этой статье я расскажу о том, как в Keras сохранять обученную сеть для последующего использования.
 
{% include youtube-player.html id="I7NeJ_pZoAA" %}
 
<!--more-->

Keras позволяет сохранить два типа информации об обученной сети:

1. Архитектура сети.
2. Веса сети, подобранные в процессе обучения.

# Сохраняем архитектуру сети

Архитектура сети, то есть набор уровней, их типы, функции активации, и т.п. может быть записана в файлы формата json или yaml. Пример для записи сети в json:

```python
# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("mnist_model.json", "w")
json_file.write(model_json)
json_file.close()
```

Пример json файла для [сети распознавания рукописных цифр MNIST](/courses/nnpython-lab1):

    {
        "class_name": "Sequential", 
        "keras_version": "1.1.1", 
        "config": [
            {
            "class_name": "Dense", 
            "config": {
                "activity_regularizer": null, 
                "b_constraint": null, 
                "input_dim": 784, 
                "W_constraint": null, 
                "input_dtype": "float32", 
                "output_dim": 800, 
                "trainable": true, 
                "name": "dense_1", 
                "activation": "relu", 
                "batch_input_shape": [null, 784], 
                "b_regularizer": null, 
                "bias": true, 
                "init": "normal", 
                "W_regularizer": null}
             }, 
             {
             "class_name": "Dense", 
             "config": {
                 "activity_regularizer": null, 
                 "b_constraint": null, 
                 "input_dim": null, 
                 "name": "dense_2", 
                 "output_dim": 10, 
                 "trainable": true, 
                 "activation": "softmax", 
                 "W_constraint": null, 
                 "b_regularizer": null, 
                 "bias": true, 
                 "init": "normal", 
                 "W_regularizer": null}
             }
         ]
    }

Видно, что используется сеть типа *Sequential*, в ней есть два уровня типа *Dense*. На первом уровне 784 входа, 800 нейронов, фукнция активации *RELU*. На втором слое 10 нейронов, функция активации *Softmax*. Сеть именно такая, какую [мы создавали в программе](/courses/nnpython-lab1).

Если вы хотите использовать для сохранения сети формат yaml вместо json, то используйте метод `model.to_yaml()`:

```python
# Генерируем описание модели в формате yaml
model_yaml = model.to_yaml()
yaml_file = open("mnist_model.yml", "w")
# Записываем модель в файл
yaml_file.write(model_yaml)
yaml_file.close()
```

# Сохраняем данные о весах

Для сохранения данных о весах сети Keras использует [формат HDF5](https://support.hdfgroup.org/HDF5/). Чтобы сохранить веса, вызывайте метод `model.save_weights()`:

```python
model.save_weights("mnist_model.h5")
```

HDF5 - бинарный формат, для его просмотра нам потребуются специальные утилиты, например, [HDFView](https://support.hdfgroup.org/products/java/hdfview/).

![Просмотр весов сети в HDFView](/assets/dl/weights_hdfview.jpg)

В левой части экрана мы видим слои сети (два слоя dense_1 и dense_2), а в правой - значения весов для слоя dense_1.

Может возникнуть вопрос, зачем отдельно сохранять архитектуру сети и значения весов. Так сделано для того, чтобы можно было загрузить значения весов в сеть с другой архитектурой. Такой подход используется, например, при совмещении обучения без учителя и с учителем. На первом этапе выполняется обучения без учителя с использованием автокодировщика, глубокой сети доверия или другого метода. Затем полученные веса загружаются в сеть другой архитектуры, которая дообучается стандартным подходом обучения с учителем с помощью метода обратного распространения ошибки. Совмещение двух способов позволяет обучать сеть в случае, когда мало размеченных данных для обучения. Подробнее об этом я напишу в одной из следующих статей.

# Загружаем обученную сеть

Для загрузки сети, как и для сохранения, нам необходимо выполнить две операции:

1. Загрузить данные об архитектуре сети.
2. Загрузить данные о весах.

Пример программы загрузки нейронной сети:

```python
# Загружаем данные об архитектуре сети из файла json
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("mnist_model.h5")
```

Перед использованием модели, ее обязательно нужно скомпилировать:

```python
# Компилируем модель
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
# Проверяем модель на тестовых данных
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("Точность модели на тестовых данных: %.2f%%" % (scores[1]*100))
```

# Ссылки

1. [Keras FAQ: How can I save a Keras model?](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)
2. [Save and Load Your Keras Deep Learning Models](http://machinelearningmastery.com/save-load-keras-deep-learning-models/).
3. Полный текст программ из статьи - [https://github.com/sozykin/dlpython_course](https://github.com/sozykin/dlpython_course).

