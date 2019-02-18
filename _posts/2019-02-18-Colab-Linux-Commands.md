---
layout: post
title:  "Команды Linux на платформе Google Colaboratory"
date:   2019-02-18 07:30:00 +0500
categories: deep_learning
comments: true
---
{% include youtube-player.html id="1Q_MF-IXzDQ" %}

[Google Colaboratory](/deep_learning/2018/04/04/Google-Colaboratory-for-Deep-Learning.html) - это бесплатная облачная платформа от Google, где установлено много популярных библиотек для машинного обучения, а также есть GPU. Платформа удобна для изучения машинного обучения и нейронных сетей, т.к. на ней вы сразу получаете готовую к использованию среду для машинного обучения, за которую не нужно платить.

Для большинства задач в Colaboratory достаточно средств Python, но иногда все-таки приходится использовать команды операционной системы. Например, для загрузки данных или установки пакетов Python. Код ноутбуков, которые мы создаем на платформе Google Colaboratory, запускается на виртуальной машине с Linux в облаке Google. Если вы привыкли работать с Wndows, то использование команд Linux на первом этапе может вызвать затруднение. В этой статье я расскажу, как использовать команды Linux в ноутбуках Colaboratory. Все команды есть в [ноутбуке с полным текстом примеров кода](https://colab.research.google.com/drive/1vFGZ2nDS0ukNGXPL-0avK097afYQILyq).

<!--more-->

## Основы использования командной строки в Colab

Комады Linux можно писать в ячейках ноутбуков Colaboratory, также как и команды Python. Но команды Linux должны начинаться с восклицательного знака (!). Именно по наличию восклицательного знака в начале строки определяется, что это не код Python, а команда операционной системы. 

Пример:

![Команда Linux в ноутбуке Colab](/assets/dl/colab_linux_command.png)

Дальше буду писать команды текстом, чтобы вы могли их копировать в свои ноутбуки:

```
!ls
```
```
sample_data
```

`ls` - команда для получения списка файлов в каталоге. Мы можем видеть, что в текущем каталоге есть всего один файл - `sample_data`. Но это на самом деле не файл, а каталог. Давайте посмотрим, что находится в нем.

```
!ls sample_data
```
```
anscombe.json		      mnist_test.csv
california_housing_test.csv   mnist_train_small.csv
california_housing_train.csv  README.md
```

Если команде `ls` в качестве параметра указать имя каталога, то будет выведен список файлов в этом каталоге. Как мы видим, в каталоге файлы в форматах JSON, csv и Markdown.

Каталог `sample_data`, как можно понять из названия, содержит демонстрационные данные. Этот каталог и данные в нем создаются автоматически при запуске виртуальной машины.

## Просмотр файлов

Как узнать, что находится в файлах с демонстрационными данными? Можно написать код на Python, но быстрее использовать команды Linux. Команда `head` выводит первые строки указанного ей файла. 

```
!head sample_data/california_housing_train.csv
```
```
"longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"
-114.310000,34.190000,15.000000,5612.000000,1283.000000,1015.000000,472.000000,1.493600,66900.000000
-114.470000,34.400000,19.000000,7650.000000,1901.000000,1129.000000,463.000000,1.820000,80100.000000
-114.560000,33.690000,17.000000,720.000000,174.000000,333.000000,117.000000,1.650900,85700.000000
-114.570000,33.640000,14.000000,1501.000000,337.000000,515.000000,226.000000,3.191700,73400.000000
-114.570000,33.570000,20.000000,1454.000000,326.000000,624.000000,262.000000,1.925000,65500.000000
-114.580000,33.630000,29.000000,1387.000000,236.000000,671.000000,239.000000,3.343800,74000.000000
-114.580000,33.610000,25.000000,2907.000000,680.000000,1841.000000,633.000000,2.676800,82400.000000
-114.590000,34.830000,41.000000,812.000000,168.000000,375.000000,158.000000,1.708300,48500.000000
-114.590000,33.610000,34.000000,4789.000000,1175.000000,3134.000000,1056.000000,2.178200,58400.000000
```

В первой строке файла файла `california_housing_train.csv` заголовок, а в остальных данные, разделенные запятыми.

По-умолчанию команда head выводит 10 строк из файла. Можно показать любое количество срок, использую параметр `-n`. Давайте выведем 15 строк:

```
!head -n 15 sample_data/california_housing_train.csv
```
```
"longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"
-114.310000,34.190000,15.000000,5612.000000,1283.000000,1015.000000,472.000000,1.493600,66900.000000
-114.470000,34.400000,19.000000,7650.000000,1901.000000,1129.000000,463.000000,1.820000,80100.000000
-114.560000,33.690000,17.000000,720.000000,174.000000,333.000000,117.000000,1.650900,85700.000000
-114.570000,33.640000,14.000000,1501.000000,337.000000,515.000000,226.000000,3.191700,73400.000000
-114.570000,33.570000,20.000000,1454.000000,326.000000,624.000000,262.000000,1.925000,65500.000000
-114.580000,33.630000,29.000000,1387.000000,236.000000,671.000000,239.000000,3.343800,74000.000000
-114.580000,33.610000,25.000000,2907.000000,680.000000,1841.000000,633.000000,2.676800,82400.000000
-114.590000,34.830000,41.000000,812.000000,168.000000,375.000000,158.000000,1.708300,48500.000000
-114.590000,33.610000,34.000000,4789.000000,1175.000000,3134.000000,1056.000000,2.178200,58400.000000
-114.600000,34.830000,46.000000,1497.000000,309.000000,787.000000,271.000000,2.190800,48100.000000
-114.600000,33.620000,16.000000,3741.000000,801.000000,2434.000000,824.000000,2.679700,86500.000000
-114.600000,33.600000,21.000000,1988.000000,483.000000,1182.000000,437.000000,1.625000,62000.000000
-114.610000,34.840000,48.000000,1291.000000,248.000000,580.000000,211.000000,2.157100,48600.000000
-114.610000,34.830000,31.000000,2478.000000,464.000000,1346.000000,479.000000,3.212000,70400.000000
```

Если хотим просмотреть весь файл, то можно использовать команду `cat`:

```
!cat sample_data/README.md
```
```
This directory includes a few sample datasets to get you started.

* `california_housing_data*.csv` is California housing data from the 1990 US
  Census; more information is available at:
  https://developers.google.com/machine-learning/crash-course/california-housing-data-description

* `mnist_*.csv` is a small sample of the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database), which is described
  at: http://yann.lecun.com/exdb/mnist/

* `anscombe.json` contains a copy of [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet);
  it was originally described in

      Anscombe, F. J. (1973). 'Graphs in Statistical Analysis'. American Statistician. 27 (1): 17-21. JSTOR 2682899.

  and our copy was prepared by the [vega_datasets library](https://github.com/altair-viz/vega_datasets/blob/4f67bdaad10f45e3549984e17e1b3088c731503d/vega_datasets/_data/anscombe.json).
```

Выводить весь файл стоит только если файл небольшой и мы сможем его посмотреть. Но как узнать, большой файл, или нет? Команда wc -l показывает количество срок в файле.

```
!wc -l sample_data/README.md
!wc -l sample_data/california_housing_train.csv
```
```
17 sample_data/README.md
17001 sample_data/california_housing_train.csv
```

В файле `README.md` 17 строк, а в `california_housing_train.csv` - 17001. Выводить целиком файл `california_housing_train.csv` явно не стоит, он слишком большой.

В одной ячейке можно указывать не одну команду Linux, а несколько, как в примере с двумя командами `wc -l`.

## Каталоги и пути к файлам

В Linux для разделения имен файлов, каталогов и подкаталогов используется прямой слеш - `/` (в Windows для этой цели используется обратный слеш `\`). Python часто возволяет писать пути к файлам с любым типом слеша и автоматически конвертирует в понятный текущей операционной системе вариант. Но Linux так делать не умеет. Если вы попробуете использовать обратный слеш, как в Windows, то будет ошибка:

```
!head sample_data\california_housing_train.csv
```
```
head: cannot open 'sample_datacalifornia_housing_train.csv' for reading: No such file or directory
```

Правильный вариант:

```
!head sample_data/california_housing_train.csv
```
```
"longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"
-114.310000,34.190000,15.000000,5612.000000,1283.000000,1015.000000,472.000000,1.493600,66900.000000
-114.470000,34.400000,19.000000,7650.000000,1901.000000,1129.000000,463.000000,1.820000,80100.000000
-114.560000,33.690000,17.000000,720.000000,174.000000,333.000000,117.000000,1.650900,85700.000000
-114.570000,33.640000,14.000000,1501.000000,337.000000,515.000000,226.000000,3.191700,73400.000000
-114.570000,33.570000,20.000000,1454.000000,326.000000,624.000000,262.000000,1.925000,65500.000000
-114.580000,33.630000,29.000000,1387.000000,236.000000,671.000000,239.000000,3.343800,74000.000000
-114.580000,33.610000,25.000000,2907.000000,680.000000,1841.000000,633.000000,2.676800,82400.000000
-114.590000,34.830000,41.000000,812.000000,168.000000,375.000000,158.000000,1.708300,48500.000000
-114.590000,33.610000,34.000000,4789.000000,1175.000000,3134.000000,1056.000000,2.178200,58400.000000
```

До сих пор мы просматривали файлы в текущем каталоге. Но как узнать, где именно в файловой системе находится этот каталог? Для этой цели используется команда `pwd`:

```
!pwd
```
```
/content
```

Путь к текущему каталогу - `/content`. В Linux нет разных дисков, все пути в файловой системе начинаются от так называемого корневого каталога, который обозначается прямым слешем - `/`.  Давайте посмотрим, какие есть каталоги верхнего уровня:

```
!ls /
```
```
bin   content  dev  home  lib32  media	opt   root  sbin  swift  tmp	usr
boot  datalab  etc  lib   lib64  mnt	proc  run   srv   sys	 tools	var
```

Наиболее важные каталоги верхнего уровня:
- `bin`, `sbin` - исполняемые файлы.
- `content` - каталог для данных, его мы уже рассматривали выше.
- `home` - каталог для файлов пользователей ("домашний" каталог).
- `root` - домашний каталог для пользователя `root` (аналог Администратора в Linux).
- `lib`, `lib32`, `lib64` - библиотеки.
- `tmp` - каталог для временных файлов.
- `etc` - каталог с конфигурационными файлами.

Потратьте некоторое время чтобы посмотреть содержимое этих каталогов.

В большей части этих каталогов мы не будем ничего менять. Нам нужно будет загружать файлы данных в каталог `content` и выполнять некоторые настройки в домашнем каталоге пользователя `root`.

## Копирование файлов и создание каталогов

Предположим, что мы решили создать проект, который использует демонстрационные данные Сalifornia Housing. Чтобы случайно не испортить исходные файлы, мы создадим отдельный каталог и скопируем файлы туда.

Создание каталога выполняется командой `mkdir`:

```
!mkdir my_project
```

Проверим, что каталог создался с помощью уже известной нам команды `ls`:

```
!ls
```
```
my_project  sample_data
```

В текущем каталоге появился новый каталог `my_project`. Теперь скопируем в него демонстрационные данные. Для этого используется команда `cp`:

```
!cp sample_data/california_housing_test.csv my_project
!cp sample_data/california_housing_train.csv my_project
```

Первый параметр этой команды - имя файла, который нужно скопировать, а второй - имя файла или каталога, куда нужно скопировать.

Проверим, что файлы скопировались:

```
!ls my_project/
```
```
california_housing_test.csv  california_housing_train.csv
```

В новом каталоге файлы с демонстрационными данными есть. Теперь можно работать с ними не опасаясь что-нибудь испортить.

## Загрузка данных из интернет

Чаще всего нужные нам данные выложены в интернет и мы загружаем их на локальный компьютер через браузер. Но ноутбук Colaboratory выполняется в виртуальной машине в облаке Google, поэтому загружать файлы нужно на эту виртуальную машину. В Linux для загрузки файлов из интернет используется команда `wget`. Для примера давайте загрузим [данные о работе такси в Нью-Йорке](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

```
!wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-06.csv
```
```
--2019-02-06 03:09:51--  https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2018-06.csv
Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.85.213
Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.85.213|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 769389923 (734M) [text/csv]
Saving to: ‘yellow_tripdata_2018-06.csv’

yellow_tripdata_201 100%[===================>] 733.75M  36.7MB/s    in 21s     

2019-02-06 03:10:12 (35.5 MB/s) - ‘yellow_tripdata_2018-06.csv’ saved [769389923/769389923]
```

После завершения загрузки проверим, что файл появился в текущем каталоге:

```
!ls
```
```
my_project  sample_data  yellow_tripdata_2018-06.csv
```

Файл появился. Давайте просмотрим его начало:

```
!head yellow_tripdata_2018-06.csv
```
```
VendorID,tpep_pickup_datetime,tpep_dropoff_datetime,passenger_count,trip_distance,RatecodeID,store_and_fwd_flag,PULocationID,DOLocationID,payment_type,fare_amount,extra,mta_tax,tip_amount,tolls_amount,improvement_surcharge,total_amount

1,2018-06-01 00:15:40,2018-06-01 00:16:46,1,.00,1,N,145,145,2,3,0.5,0.5,0,0,0.3,4.3
1,2018-06-01 00:04:18,2018-06-01 00:09:18,1,1.00,1,N,230,161,1,5.5,0.5,0.5,1.35,0,0.3,8.15
1,2018-06-01 00:14:39,2018-06-01 00:29:46,1,3.30,1,N,100,263,2,13,0.5,0.5,0,0,0.3,14.3
1,2018-06-01 00:51:25,2018-06-01 00:51:29,3,.00,1,N,145,145,2,2.5,0.5,0.5,0,0,0.3,3.8
1,2018-06-01 00:55:06,2018-06-01 00:55:10,1,.00,1,N,145,145,2,2.5,0.5,0.5,0,0,0.3,3.8
1,2018-06-01 00:09:00,2018-06-01 00:24:01,1,2.00,1,N,161,234,1,11.5,0.5,0.5,2.55,0,0.3,15.35
1,2018-06-01 00:02:33,2018-06-01 00:13:01,2,1.50,1,N,163,233,1,8.5,0.5,0.5,1.95,0,0.3,11.75
1,2018-06-01 00:13:23,2018-06-01 00:16:52,1,.70,1,N,186,246,1,5,0.5,0.5,1.85,0,0.3,8.15
```

Как мы и ожидали, в файле данные о поездках такси в Нью-Йорке в формате `csv`.

## Архивы

Для качественного обучения моделей как правило, нужно много данных. А если данных много, то их обычно запаковывают в архивы, чтобы занимали меньше места. Давайте посмотрим, как можно под Linux распаковывать архивы.

### Архивы zip

Начнем с простого - загрузим и распакуем данные в архиве `Zip`.  В качестве примера будем использовать небольшую часть набора данных  [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/#download) - фотографии людей, чьи имена начинаются на A.

```
!wget http://vis-www.cs.umass.edu/lfw/lfw-a.zip
```
```
--2019-02-06 03:12:30--  http://vis-www.cs.umass.edu/lfw/lfw-a.zip
Resolving vis-www.cs.umass.edu (vis-www.cs.umass.edu)... 128.119.244.95
Connecting to vis-www.cs.umass.edu (vis-www.cs.umass.edu)|128.119.244.95|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 15167852 (14M) [application/zip]
Saving to: ‘lfw-a.zip’

lfw-a.zip           100%[===================>]  14.46M  6.07MB/s    in 2.4s    

2019-02-06 03:12:33 (6.07 MB/s) - ‘lfw-a.zip’ saved [15167852/15167852]
```

Распаковываем архив с помощью команды `unzip`:

```
!unzip lfw-a.zip
```
```
Archive:  lfw-a.zip
   creating: lfw/Aaron_Eckhart/
  inflating: lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg  
   creating: lfw/Aaron_Guiel/
  inflating: lfw/Aaron_Guiel/Aaron_Guiel_0001.jpg  
   creating: lfw/Aaron_Patterson/
  inflating: lfw/Aaron_Patterson/Aaron_Patterson_0001.jpg  
   creating: lfw/Aaron_Peirsol/
  inflating: lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg  
  inflating: lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg  
  inflating: lfw/Aaron_Peirsol/Aaron_Peirsol_0003.jpg  
  inflating: lfw/Aaron_Peirsol/Aaron_Peirsol_0004.jpg  
...
```

Команда `unzip` выводит список действий, которые она выполняет. Создается каталог `lfw`, в нем подкаталоги для каждого человека: `Aaron_Eckhart`, `Aaron_Guiel`, `Aaron_Patterson` и т.п. В каждый каталог распаковываются фотографии этого человека.

Проверяем результаты распаковки:

```
!ls lfw
```
```
Aaron_Eckhart		      Amanda_Coetzer
Aaron_Guiel		      Amanda_Marsh
Aaron_Patterson		      Amanda_Plumer
Aaron_Peirsol		      Amber_Frey
Aaron_Pena		      Amber_Tamblyn
Aaron_Sorkin		      Ambrose_Lee
...
```

В каталоге `lfw` мы видим список подкаталогов с именами знаменитостей. Давайте посмотрим, что в одном из этих каталогов:
```
!ls lfw/Angelina_Jolie
```
```
Angelina_Jolie_0001.jpg  Angelina_Jolie_0008.jpg  Angelina_Jolie_0015.jpg
Angelina_Jolie_0002.jpg  Angelina_Jolie_0009.jpg  Angelina_Jolie_0016.jpg
Angelina_Jolie_0003.jpg  Angelina_Jolie_0010.jpg  Angelina_Jolie_0017.jpg
Angelina_Jolie_0004.jpg  Angelina_Jolie_0011.jpg  Angelina_Jolie_0018.jpg
Angelina_Jolie_0005.jpg  Angelina_Jolie_0012.jpg  Angelina_Jolie_0019.jpg
Angelina_Jolie_0006.jpg  Angelina_Jolie_0013.jpg  Angelina_Jolie_0020.jpg
Angelina_Jolie_0007.jpg  Angelina_Jolie_0014.jpg
```

Фотографий Анджелины Джоли в наборе LFW достаточно много, 20 штук.

### Архивы tar

Более сложный вариант - архив в формате `tar`, сжатый архиватором `gzip`. Такие файлы имеют расширение `.tar.gz` (или `.tgz`). В Linux такой формат архивов более популярен, чем `zip`. Именно в таком виде распространяются предварительно обученные модели TensorFlow. 

Давайте загрузим модель TensorFlow для обнаружения объектов на изображениях и распакуем её. Загружаем модель с помощью `wget`:

```
!wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
```
```
--2019-02-06 03:16:38--  http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
Resolving download.tensorflow.org (download.tensorflow.org)... 74.125.206.128, 2a00:1450:400c:c04::80
Connecting to download.tensorflow.org (download.tensorflow.org)|74.125.206.128|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 672221478 (641M) [application/x-tar]
Saving to: ‘faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz’

faster_rcnn_incepti 100%[===================>] 641.08M  79.0MB/s    in 8.8s    

2019-02-06 03:16:47 (72.5 MB/s) - ‘faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz’ saved [672221478/672221478]

```

Распаковываем архив командой `tar`:
```
!tar xvfz faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
```
```
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt.index
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/checkpoint
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/pipeline.config
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt.data-00000-of-00001
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt.meta
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/saved_model/
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/saved_model/saved_model.pb
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/saved_model/variables/
faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb
```

Команде `tar` передаются следующие опции:
- `x` - нужно распаковать архив (от eXtract).
- `v` - выводить информацию о своих действиях (от verbose).
- `f` - брать данные из файла (от file). Изначально архиватор `tar` создавался для работы с магнитными лентами, его название расшифровывается как tape archiver. Поэтому `tar` без параметра `f` будет пытаться читать данные с ленты.
- `z` - перед извлечением файлов выполнить декомпрессию с помощью `gzip`.

В процессе работы команда `tar` пишет о файлах, которые она извлекает. Проверяем распакованное содержимое архива:

```
!ls faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28
```
```
checkpoint			model.ckpt.index  saved_model
frozen_inference_graph.pb	model.ckpt.meta
model.ckpt.data-00000-of-00001	pipeline.config
```

Все файлы из архива на месте.

## Установка пакетов

### Пакеты Python

На Colab уже установлено много полезных пакетов Python для машинного обучения и других задач. Но если вам понадобился какой-то новый пакет Python, то его можно достаточно просто установить с помощью команды `pip`. Например, установим библиотеку [`mxnet`](https://mxnet.apache.org):

```
!pip install mxnet
```
```
Collecting mxnet
  Downloading https://files.pythonhosted.org/packages/f0/2e/b26eb7273aed1945f59993b3b306442eb41684f931b5380821c39cf50a31/mxnet-1.3.1-py2.py3-none-manylinux1_x86_64.whl (27.1MB)
    100% |████████████████████████████████| 27.1MB 1.7MB/s 
Collecting graphviz<0.9.0,>=0.8.1 (from mxnet)
  Downloading https://files.pythonhosted.org/packages/53/39/4ab213673844e0c004bed8a0781a0721a3f6bb23eb8854ee75c236428892/graphviz-0.8.4-py2.py3-none-any.whl
Requirement already satisfied: numpy<1.15.0,>=1.8.2 in /usr/local/lib/python3.6/dist-packages (from mxnet) (1.14.6)
Collecting requests>=2.20.0 (from mxnet)
  Downloading https://files.pythonhosted.org/packages/7d/e3/20f3d364d6c8e5d2353c72a67778eb189176f08e873c9900e10c0287b84b/requests-2.21.0-py2.py3-none-any.whl (57kB)
    100% |████████████████████████████████| 61kB 23.7MB/s 
Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests>=2.20.0->mxnet) (1.22)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests>=2.20.0->mxnet) (2018.11.29)
Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests>=2.20.0->mxnet) (2.6)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests>=2.20.0->mxnet) (3.0.4)
Installing collected packages: graphviz, requests, mxnet
  Found existing installation: graphviz 0.10.1
    Uninstalling graphviz-0.10.1:
      Successfully uninstalled graphviz-0.10.1
  Found existing installation: requests 2.18.4
    Uninstalling requests-2.18.4:
      Successfully uninstalled requests-2.18.4
Successfully installed graphviz-0.8.4 mxnet-1.3.1 requests-2.21.0
```

Как можно видеть, выполнилась установка не только пакета `mxnet`, но и других пакетов, которые необходимы ему для работы. 

## Пакеты операционной системы

Иногда может понадобиться установка пакетов операционной системы. Например, библиотеки для работы с архивами в разных форматах [libarchive](https://www.libarchive.org/). Системные пакеты устанавливаются командой `apt-get`:

```
!apt-get install libarchive-dev -y
```
```
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following NEW packages will be installed:
  libarchive-dev
0 upgraded, 1 newly installed, 0 to remove and 3 not upgraded.
Need to get 470 kB of archives.
After this operation, 1,621 kB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu bionic-updates/main amd64 libarchive-dev amd64 3.2.2-3.1ubuntu0.2 [470 kB]
Fetched 470 kB in 1s (681 kB/s)
Selecting previously unselected package libarchive-dev:amd64.
(Reading database ... 111313 files and directories currently installed.)
Preparing to unpack .../libarchive-dev_3.2.2-3.1ubuntu0.2_amd64.deb ...
Unpacking libarchive-dev:amd64 (3.2.2-3.1ubuntu0.2) ...
Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
Setting up libarchive-dev:amd64 (3.2.2-3.1ubuntu0.2) ...
```

## Работа с Git

В Colaboratory можно использовать обычные команды [`git`](https://git-scm.com/). Например, склонируем репозиторий с моделями [`TensorFlow`](https://github.com/tensorflow/models):

```
!git clone https://github.com/tensorflow/models.git
```
```
Cloning into 'models'...
remote: Enumerating objects: 17, done.
remote: Counting objects: 100% (17/17), done.
remote: Compressing objects: 100% (17/17), done.
remote: Total 24450 (delta 1), reused 0 (delta 0), pack-reused 24433
Receiving objects: 100% (24450/24450), 563.42 MiB | 28.85 MiB/s, done.
Resolving deltas: 100% (14529/14529), done.
Checking out files: 100% (2768/2768), done.
```

Проверяем результаты клонирования:

```
!ls models
```
```
AUTHORS     CONTRIBUTING.md    LICENSE	 README.md  samples    WORKSPACE
CODEOWNERS  ISSUE_TEMPLATE.md  official  research   tutorials
```

Все необходимые каталоги на месте. Просмотрим файл `README.md`:

```
!cat models/README.md
```
```
# TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)
```

## Ресурсы виртуальной машины Linux в Colaboratory

Наверняка вам интересно узнать, насколько мощную виртуальную машину дает Google. Это тоже можно сделать с помощью команд Linux.

### Какой Linux используется в Colaboratory?

Узнать, какой дистрибутив Linux и какая его версия используется, можно следующей командой:

```
!lsb_release -a  
```
```
No LSB modules are available.
Distributor ID:	Ubuntu
Description:	Ubuntu 18.04.1 LTS
Release:	18.04
Codename:	bionic
```

В Colab используется Linux Ubuntu 18.04.

### Сколько процесоров и памяти?

Количество процессоров можно узнать, прочитав данные из файловой системы специального назначения `/proc`:

```
!cat /proc/cpuinfo
```
```
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 63
model name	: Intel(R) Xeon(R) CPU @ 2.30GHz
stepping	: 0
microcode	: 0x1
cpu MHz		: 2300.000
cache size	: 46080 KB
physical id	: 0
siblings	: 2
core id		: 0
cpu cores	: 1
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single pti ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt arat arch_capabilities
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf
bogomips	: 4600.00
clflush size	: 64
cache_alignment	: 64
address sizes	: 46 bits physical, 48 bits virtual
power management:

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 63
model name	: Intel(R) Xeon(R) CPU @ 2.30GHz
stepping	: 0
microcode	: 0x1
cpu MHz		: 2300.000
cache size	: 46080 KB
physical id	: 0
siblings	: 2
core id		: 0
cpu cores	: 1
apicid		: 1
initial apicid	: 1
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single pti ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid xsaveopt arat arch_capabilities
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf
bogomips	: 4600.00
clflush size	: 64
cache_alignment	: 64
address sizes	: 46 bits physical, 48 bits virtual
power management:
```

В выводе команды мы можем видеть два процессора `Intel(R) Xeon(R) CPU @ 2.30GHz`. Но на самом деле это не два отдельных процессора, а два ядра. Производитель процессоров Intel, тактовая частота 2,3 ГГц.

Если есть желание, попробуйте подробнее разобрать информацию из `cpuinfo`, там можно найти много интересного. Рекомендую начать с раздела `bugs`.

### Память

Узнать объем доступной оперативной памяти можно командой `free`:

```
!free -h
```
```
              total        used        free      shared  buff/cache   available
Mem:            12G        431M         10G        900K        1.7G         12G
Swap:            0B          0B          0B
```

Всего памяти 12 ГБ, из них свободно 10ГБ.

### Какой GPU используется?

Если вы подключили GPU в разделе Runtime -> Change Runtime Type -> Hardware accelerator, то модель GPU можно посмотреть следующей командой: 

```
!cat /proc/driver/nvidia/gpus/0000:00:04.0/information
```
```
Model: 		 Tesla K80
IRQ:   		 33
GPU UUID: 	 GPU-c6398fd0-9366-8e15-5d6a-369da852f85b
Video BIOS: 	 80.21.25.00.02
Bus Type: 	 PCI
DMA Size: 	 40 bits
DMA Mask: 	 0xffffffffff
Bus Location: 	 0000:00:04.0
Device Minor: 	 0
Blacklisted:	 No
```

В Colab используется GPU NVIDIA Tesla K80. Не самая последняя модель, но достаточно производительная.

Если у вас подключен GPU, но указанная выше команда выдает ошибку, то, возможно, путь к GPU у вас отличается. Чтобы узнать правильный путь, выполните команду:

```
!ls /proc/driver/nvidia/gpus/
```
```
0000:00:04.0
```

Значение из вывода этой команды вставьте в путь к GPU из предыдущей команды.

### Под каким пользователем Linux выполняются команды в Colaboratory?

Узнать, от имени какого пользователя выполняются команды, можно с помощью команды `whoami` (от английского "Кто я?"):

```
!whoami
```
```
root
```

Имя пользователя - `root`, это суперпользователь, аналог Администратора в Windows. Домашний каталог пользователя - `/root`. 

Учтите, что пользователь `root` имеет права на выполнение любых действий в Linux, в том числе деструктивных. Поэтому будьте осторожны при выполнении команд, т.к. если вы сделаете что-то неправильно, то можете, например, удалить все файлы на виртуальной машине. С другой стороны, ничего страшного в этом случае не произойдет. Просто перезапустите виртуальную машину, и она вернется к состоянию по умолчанию, с которого мы начинали в этой статье.

## Итоги

Мы научились выполнять команды Linux в ноутбуках Google Colaboratory. Команды Linux отличаются от Python тем, что в начале ячейки указывается восклицательный знак - `!`.

Чаще всего команды Linux выполняются для загрузки файлов, работы с файлами и архивами, а также установки новых пакетов Python и Linux. Теперь вы знаете все необходимые для этого команды. 

Кроме указанных в статье команд, в ячейках Colab можно выполнять и другие команды Linux. Для этого ячейка должна начинаться с `!`. Но не забудьте, что вы работаете с правами суперпользователя и можете случайно сломать виртуальную машину, если не до конца понимаете, что делаете.

Пишите в комментариях, какие еще команды Linux вам кажутся полезными в Colaboratory.

## Полезные ссылки

1. [Полный текст ноутбука с примерами использования команд Linux в Colab](https://colab.research.google.com/drive/1vFGZ2nDS0ukNGXPL-0avK097afYQILyq).
2. [Общее описание работы с Google Colaboratory](/deep_learning/2018/04/04/Google-Colaboratory-for-Deep-Learning.html).
3. [Использование бесплатного тензорного процессора TPU для обучения нейронных сетей в Google Colaboratory](https://habr.com/ru/post/428117/).

