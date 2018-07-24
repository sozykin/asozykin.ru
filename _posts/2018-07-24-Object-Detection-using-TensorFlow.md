---
layout: post
title:  "Поиск объектов на изображениях"
date:   2018-07-24 10:00:00 +0500
categories: deep_learning
comments: true
---
{% include youtube-player.html id="1KzKqxx-s-Q" %}

Одна из востребованных задач компьютерного зрения, которая может быть решена с помощью глубоких нейронных сетей -- это поиск на изображении объектов заданного типа. В этой статье я расскажу, как можно решить такую задачу с помощью предварительно обученных нейронных сетей из [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Эта система является частью TensorFlow и распространяется бесплатно, также как и сама TensorFlow. Полный текст кода из статьи можно найти в [репозитории курса на github](https://github.com/sozykin/dlpython_course/blob/master/computer_vision/object_detection/object_detection_api.ipynb), а также в [ноутбуке на Colaboratory](https://colab.research.google.com/drive/1EQ3Lt_ez-oKTtVMebh6Tm3XSyPPOHAf3) (*это бесплатная облачная платформа от Google, где уже установлены библиотеки глубокого обучения и есть GPU, ноутбук можно выполнить прямо на этой платформе*).

<!--more-->

## Классификация изображений и обнаружение объектов

Мы уже рассматривали, как применять нейронные сети для решения более простой задачи -- [классификации объектов на изображении](http://localhost:4000/courses/nnpython-lab2). Это делается достаточно просто с помощью сверточных нейронных сетей. Однако для классификации необходимо, чтобы на изображении был всего один объект. Но что делать, если на вашей картинке несколько объектов? В этом случае нам нужно решить две задачи:
1. *Сегментация* - выделение участков изображения, которые относятся к разным объектам.
2. *Классификация*, то есть определение типа объекта, для каждого выделенного сегмента отдельно.

Суммарно эти две задачи и составляют *обнаружение объектов*. В результате работы нейронной сети мы должны получить регионы изображения, в которых нейронная сеть нашла интересующие нас объекты, а также тип объекта в каждом регионе.

## Установка TensorFlow Object Detection API

TensorFlow Object Detection API не входит в основной комплект TensorFlow, а требует отдельной установки. Первый шаг - это скачивание [репозитория с моделями TensorFlow](https://github.com/tensorflow/models). Object Detection API находится в каталоге `research/object_detection`. Инструкции по установке для [Linux есть в самом дистрибутиве](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md), а для [Windows в отдельной статье](https://medium.com/@rohitrpatil/how-to-use-tensorflow-object-detection-api-on-windows-102ec8097699).

Особенность установки в том, что TensorFlow Object Detection API использует для настройки параметров нейронных сетей [Google Protocol Buffers](https://developers.google.com/protocol-buffers/), его нужно установить и скомпилировать настройки с помощью `protoc`. В Windows у меня заработала версия Protocol Buffers 3.4, с более новыми были ошибки. О возможности этих ошибках написано в инструкциях по установке. 

## Набор данных COCO

Большая часть моделей в TensorFlow Object Detection API обучены на [наборе данных COCO](http://cocodataset.org/) (Common Objects in Context, обычные объекты в контексте). Это открытый набор данных, который содержит изображения 90 классов объектов. Но в отличие от набора данных CIFAR-10 и ему подобных, на каждом изображении набора данных COCO не один объект, а несколько. Именно эти и отображено в названии набора данных: объекты в контексте. Вот пример одного из изображений:

![Пример изображения из набора данных COCO](/assets/dl/coco.jpg)

Здесь мы видим целую улицу, на которой выделены интересующие нас объекты: автобусы, машины, светофоры и люди. Для каждого объекта есть разметка, которая указывает, где находится объект, а также указан класс объекта. Другие примеры можно посмотреть на [сайте COCO](http://cocodataset.org/#explore). Именно на таких изображениях и обучались нейронные сети из TensorFlow Object Detection API. 

## Загружаем предварительно обученную модель 

TensorFlow Object Detection API содержит большое количество предварительно обученных моделей для поиска объектов на изображениях. Скачать модели можно со страницы [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Одну модель сделать не получилось, т.к. разработчики пытаются обеспечить баланс между качеством и скоростью работы сети. Вот фрагмент таблицы с описанием моделей:

![Модели обнаружения объектов в TensorFlow Object Detection API](/assets/dl/coco_trained_models.jpg)

Для каждой модели указано время работы на GeForce GTX TITAN X и точность обнаружения объектов (метрика mAP, чем она выше, тем лучше). Самые простые модели, предназначенные для мобильных устройств, работают очень быстро, но обеспечивают небольшую точность. Сложные модели точнее, но работают медленнее. Самая точная модель `faster_rcnn_nas` (не поместилась на скриншоте), дает метрику точности 43, но работает 1833 мс, что очень много. Выберем компромиссный вариант, модель `ssd_resnet_50_fpn_coco`, которая работает 76 мс и обеспечивает точность 35.

Модель нужно загрузить на свой компьютер и распаковать. В архиве с моделью есть несколько файлов, но нас будет интересовать только один -- `frozen_inference_graph.pb`. Именно это в терминологии TensorFlow и есть предварительно обученная модель, которую можно использовать для предсказаний (inference).

## Поиск объектов на изображениях

После установки TensorFlow Object Detection API и загрузки предварительно обученной модели все готово для создания программы поиска объектов на изображениях. Начнем с импорта нужных нам модулей Python:

```python
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
%matplotlib inline
```

Мы подключаем модули для вычислений и визуализации (`numpy`, `PIL`, `matplotlib`), обычную библиотеку `tensorflow` и модули из Object Detection API: `object_detection.utils` и `utils`. Если у вас выдается ошибка при подключении модулей Object Detection API, проверьте, добавили ли вы пути к этой библиотеке при установке в переменную окружения `PYTHONPATH`. Как это сделать написано в [инструкции по установке](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

### Загружаем предварительно обученную модель в оперативную память

Чтобы использовать загруженную ранее предварительно обученную модель, ее необходимо загрузить в память и создать на ее основе `tf.Graph()`:

```python
# Путь к файлу с моделью, который вы сохранили на предыдущем этапе
model_file_name = 'ssd_resnet50_v1_fpn/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(model_file_name, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
```

### Загружаем метки классов

Предварительно обученная модель на выходе выдает номер класса объекта из набора данных COCO. Мы же хотим видеть не номера классов, а названия объектов. Для этого загрузим названия из файла с метками классов:

```python
# Путь к файлу с метками классов в каталоге с моделями TensorFlow
label_map_file_name = 'models/research/object_detection/data/mscoco_label_map.pbtxt' 
label_map = label_map_util.load_labelmap(label_map_file_name)
categories = label_map_util.convert_label_map_to_categories(label_map, 
                                                            max_num_classes=90, 
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
```

`category_index` -- это словарь, который содержит номера классов и соответствующие им названия объектов. Вот фрагмент этого словаря:

```
1: {'id': 1, 'name': 'person'},
2: {'id': 2, 'name': 'bicycle'},
3: {'id': 3, 'name': 'car'},
4: {'id': 4, 'name': 'motorcycle'},
5: {'id': 5, 'name': 'airplane'},
...
```

Класс с номером 1 -- это человек, с номером 2 -- велосипед, с номером 3 -- машина и т.д.

### Загружаем изображение для поиска объектов

Начнем наши эксперименты с простого изображения с соревнования [Kaggle по распознаванию кошек и собак](https://www.kaggle.com/c/dogs-vs-cats). Загрузим изображение с диска в оперативную память и преобразуем его в массив `numpy`:

```python
image_file_name = 'woof_meow.jpg'
image = Image.open(image_file_name)
(im_width, im_height) = image.size
image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
plt.figure(figsize=(12, 8))
plt.imshow(image_np)
```

![Картинка кота и собаки](/assets/dl/cat_and_dog.png)

На картинке есть собака и кот, простая классификация с такой картинкой не справится.


### Запускаем поиск объектов

Поиск объектов на изображении, как обычно в TensorFlow, требует достаточно много кода. Большая часть этого кода готовит в нужном формате входные или выходные данные. Приведу код почти без комментариев. Если интересно, как он работает, то можно разобраться по инструкциям на сайте [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

```python
# Используем модель (граф TensorFlow), которую ранее загрузили в память
with detection_graph.as_default():
    # Все операции в TensorFlow выполняются в сессии
    with tf.Session() as sess:
      # Готовим операции и входные данные
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Запуск поиска объектов на изображении
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image_np, 0)})

      # Преобразуем выходные тензоры типа float32 в нужный формат
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
```

В словаре `output_dict` мы получаем информацию о найденных объектах: их границах, классе, и вероятности принадлежности к этому классу.

### Визуализация результатов поиска

Для визуализации найденных объектов воспользуемся методом `vis_util.visualize_boxes_and_labels_on_image_array`. Как понятно из названия, он рисует границы объектов и метки их классов на изображении в массиве `numpy`. 

```python
vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
plt.figure(figsize=(24, 16))
plt.imshow(image_np)
```

Результаты работы:

![Картинка распознанных кота и собаки](/assets/dl/cat_and_dog_detected.png).

TensorFlow Object Detection API удалось правильно найти на картинке собаку (вероятность 83%) и кота (вероятность 77%).

## Тестируем на картинке с улицей

Кот и собака достаточно простая картинка, интереснее попробовать что-то более сложное. Например, картинку с видом улицы:

![Изображение улицы](/assets/dl/street.png).

Распознавать объекты на таких типах изображений полезно для самоуправляемых автомобилей. Хотелось бы, чтобы такой автомобиль знал, где находятся другие автомобили, видел светофоры, пешеходов и другие важные объекты. 

Сцена с улицей значительно сложнее, поэтому для качественных результатов будем использовать более сложную модель `faster_rcnn_inception_resnet_v2_atrous_coco`. Вот результаты ее работы:

![Изображение улицы](/assets/dl/street_detected.png).

Видно, что сеть нашла пешеходов (особенно радует, что замечены все пешеходы, которые переходят дорогу), светофоры, машины (часть из них определилась как `car` -- легковой автомобиль, а часть как `truck` -- грузовик) и автобусы.

## Ищем пиксельные маски объектов

TensorFlow Object Detection API может находить объекты с более высокой точностью, чем прямоугольник, в границах которого находится объект. Вместо координат прямоугольника может выдаваться писсельная маска. Для этого нужно использовать модели из [TensoFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), у которых в `Outputs` указано значение `Masks` (маски). 

Для примера загрузим модель `mask_rcnn_inception_resnet_v2_atrous_coco`, которая умеет находить пиксельные маски объекта. Код менять не нужно, найденные маски будут содержатся в словаре `output_dict`, ключ `detection_masks`. Пример работы:

![Обнаружение пиксельной маски машины](/assets/dl/car_mask.png).

Видно, что нейронная сеть нашла машину (но определила ее как `truck`, грузовик). Вокруг машины нарисован прямоугольник, а также определено более точное положение машины на фото с помощью маски.

## Итоги

Мы научились использовать TensorFlow Object Detection API для обнаружения объектов на изображениях в двух режимах: выделение прямоугольника, в котором находится объект, и пиксельной маски объекта. TensorFlow Object Detection API содержит большое количество предварительно обученных моделей, которые отличаются качеством и скоростью работы.

Если вам нужно искать свои типы объектов, то можно дообучить сети из TensorFlow Object Detection API. Пример кода есть на сайте [Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md), также есть другие примеры на [Medium](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9) и [Хабре](https://habr.com/post/358146/).

Пишите в комментариях, какие объекты вам удалось распознать и для каких задач это нужно.

## Полезные ссылки

1. [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
1. [Полный текст кода из статьи](https://github.com/sozykin/dlpython_course/blob/master/computer_vision/cats_and_dogs/cats_and_dogs_visualization.ipynb).
2. [Ноутбук на Colaboratory с кодом из статьи](https://colab.research.google.com/drive/1EQ3Lt_ez-oKTtVMebh6Tm3XSyPPOHAf3).
3. [Репозиторий с моделями TensorFLow](https://github.com/tensorflow/models). 
4. [Инструкции по установке TensorFlow Object Detection API для Linux](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
5. [Инструкции по установке TensorFlow Object Detection API для Windows](https://medium.com/@rohitrpatil/how-to-use-tensorflow-object-detection-api-on-windows-102ec8097699).
6. [Набор данных COCO](http://cocodataset.org/).
7. [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
8. [Обучение TensorFlow Object Detection API распознавать Енотов](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9).
9. [Обработка изображений: Tensorflow Object Detection API (обучение распознавать модели одежды)](https://habr.com/post/358146/).
1. Система поиска объектов на изображениях [YOLO](https://pjreddie.com/darknet/yolo/), альтернатива TensorFlow Object Detection API.
