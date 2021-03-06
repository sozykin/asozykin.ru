---
layout: post
title:  "Межсетевые экраны"
date:   2016-10-15 06:00:00 +0500
categories: computer_networks
comments: true
---
Текстовый вариант видеолекции на YouTube по межсетевым экранам.

{% include youtube-player.html id="9r6z9qggSIc" %}


Межсетевой экран - это устройство, которое отделяет разные компьютерные сети друг от друга. Другое название межсетевого экрана - брандмауэр или firewall.

<!--more-->

## Зачем нужны межсетевые экраны

При создании сетей TCP/IP в них был заложен принцип полной связности: любой компьютер в сети может соединиться с любым другим. Но тогда в сети было немного компьютеров, большая часть из которых находилась в университетах и исследовательских центрах. Сейчас ситуация кардинально поменялась: интернет стал огромной сетью с миллиардами компьютеров, которые используются не только для научных целей. В том числе в интернет появилось много злоумышленников, которые могут взломать ваш компьютер. Поэтому принцип полной связности сейчас нежизнеспособен. Наоборот, нам нужен способ отделять сети от небезопасных внешних сетей. Именно это и позволяет делать межсетевой экран.

## Как используются межсетевые экраны

Есть несколько вариантов использования межсетевых экранов. Если мы хотим защитить сеть нашей организации или домашнюю сеть, то межсетевой экран устанавливается между защищаемой сетью и интернетом. На рисунке межсетевой экран показан в виде стены с огнем (от английского firewall - огненная стена). В таких случаях, как правило, используется аппаратный межсетевой экран, который представляет собой отдельное устройство или является частью маршрутизатора.

![Межсетевой экран отделяет внутреннюю сеть от интернет](/assets/networks/firewalls/firewall1.jpg)

Другой вариант - установка программных межсетевых экранов на компьютеры, которые мы хотим защитить. Один из популярных вариантом программных межсетевых экранов - брандмауэр Windows. 

![Программный межсетевой экран](/assets/networks/firewalls/firewall2.jpg)

Программные межсетевые экраны особенно полезны для ноутбуков, с которыми вы работаете не только в защищенной сети организации, но и в других местах: других организациях, отелях, кафе, ресторанах и т.п. Здесь вы не имеете контроля над сетью и не можете обеспечить ее безопасность. 


## Как работают межсетевые экраны

Межсетевые экраны работают на сетевом и транспортном уровнях моделей OSI и TCP/IP. Они анализируют IP-адреса отправителя и получателя, а также порты транспортного уровня. 

Межсетевой экран перехватывает все пакеты, которые приходят из интернет и из внутренней сети. У межсетевого экрана есть таблица правил с описанием, какие пакеты можно передавать, а какие нельзя. Если пакет подходит под одно из разрешающих правил в таблице, он передается дальше. В противном случае пакет отбрасывается.

![Работа межсетевых экранов](/assets/networks/firewalls/firewall3.jpg)

Таблица правил выглядит следующим образом. Здесь показаны наиболее важные для понимания логики работы межсетевого экрана столбцы. В реальных межсетевых экранах таблицы могут отличаться.

| IP отправителя     | Порт отправителя | IP получателя     | Порт получателя | Протокол | Действие  |
|   :---:            |   :---:          |   :---:           |   :---:         |   :---:  |   :---:   |
| 220.10.1.0/24      |    >1024         | Вне 220.10.1.0/24 |    80           |    TCP   | Разрешить | 
| Вне 220.10.1.0/24  |    80            | 220.10.1.0/24     |   >1024         |    TCP   | Разрешить | 
| Любой              |    Любой         | Любой             | Любой           |    Любой | Запретить | 
{:.mbtablestyle}      

Основные поля в таблице - это IP-адреса и порты отправителя и получателя. Также есть поле протокол, в котором указывается используемый протокол транспортного или сетевого уровня, например, TCP, UDP или ICMP. Последнее после таблицы - действие. В нем прописывается, что межсетевой экран должен сделать с пакетом: разрешить его прохождение или запретить.

Предположим, что мы хотим существенно ограничить политику использования компьютерной сети для обеспечения безопасности. Пользователям разрешается работать с Web-сайтами в интернет, но все остальное запрещено. Для этого нам понадобились три правила, записанные в таблице выше.

**Первая строка таблицы** содержит правило, которое позволяет пакетам из внутренней сети, предназначенных для Web-серверов, проходить через межсетевой экран. Предположим, что наша внутренняя сеть имеет блок адресов *220.10.1.0/24*. Указываем этот блок в качестве IP-адреса отправителя, порт отправителя *>1024* (порты для браузеров назначаются операционной системой автоматически). IP получателя: *все, кроме 220.10.1.0/24*, порт получателя: *80* (порт на котором по умолчанию работают Web-серверы). В поле "Протокол" указываем транспортный протокол *TCP*, который используется прикладным протоколом HTTP. Действие - *разрешить*. 

**Второе правило** разрешает прохождение пакетов с ответами Web-серверов. IP-адрес отправителя: любой *вне 220.10.1.0/24*, порт отправителя: *80* (порт Web-серверов). IP-получателя: *220.10.1.0/24* (наша внутренняя сеть), порт получателя: *>1024* (автоматически назначенный порт для браузера). Протокол также *TCP*, действие *Разрешить*.  

**Третье правило** запрещает прохождение любых пакетов.

Межсетевой экран читает правила в таблице последовательно и выполняет проверку пакета по прочитанному правилу. Сначала пакет проверяется на соответствие правилу в первой строке, и если он под него подходит, то сразу же передается. Затем межсетевой экран переходит ко второму правилу, проверяет пакет, и передает его, если пакет подходит под второе правило. После этого межсетевой экран переходит к третьему правилу и отбрасывает все пакеты, которые не подошли под первые два правила. 

## Проверка флагов и состояния соединения

Правила в примере очень ограниченные, но злоумышленник все равно сможет попасть в нашу внутреннюю сеть. Для этого ему нужно сконструировать пакет, у которого порт отправителя равен *80*, IP получателя находится в нашей сети, и порт получателя *больше 1024*. Такой пакет подходит под второе правило. Наш межсетевой экран подумает, что это ответ какого-то из Web-серверов, и пропустит его. Таким образом, злоумышленники смогут подключиться к сетевым сервисам, которые работают на портах >1024. Чтобы избежать таких ситуаций, можно контролировать **флаги в заголовке TCP**, а также **факт установки соединения TCP**.

Для контроля флагов в TCP-заголовке необходимо добавить соответствующее поле в таблицу межсетевого экрана.

| IP отправителя     | Порт отправителя | IP получателя     | Порт получателя | Протокол | Флаг   | Действие  |
|   :---:            |   :---:          |   :---:           |   :---:         |   :---:  |  :---: | :---:     |
| 220.10.1.0/24      |    >1024         | Вне 220.10.1.0/24 |    80           |    TCP   | Любой  | Разрешить | 
| Вне 220.10.1.0/24  |    80            | 220.10.1.0/24     |   >1024         |    TCP   | Ack    | Разрешить | 
| Любой              |    Любой         | Любой             | Любой           |    Любой | Любой  | Запретить | 
{:.mbtablestyle}      

В первом правиле мы по-прежнему выпускаем в интернет все пакеты, которые предназначены для Web-серверов. Но во втором правиле во внутреннюю сеть пропускаем только пакеты с установленным флагом *ACK* (Acknowledgement) в заголовке TCP. Этот флаг установлен почти у всех пакетов TCP, кроме первого пакета с запросом на установку соединения (в таком пакете установлен только флаг SYN). Таким образом, злоумышленник не сможет установить соединения с сервисами нашей внутренней сети, даже если будет посылать пакеты, похожие на ответы Web-серверов.

Межсетевой экран может напрямую проверять, установлено ли TCP соединение между отправителем и получателем. Межсетевой экран перехватывает все пакеты, поэтому он видит и пакеты установки TCP соединения. Таким образом, он легко может узнать, какой компьютер из внутренней сети устанавливал соединение с компьютерами из внешней сети, и с какими именно. После того, как межсетевой экран увидел успешную процедуру установки соединения TCP (трехкратное рукопожатие), он вносит запись в **таблицу установленных соединений**.

| IP отправителя     | Порт отправителя | IP получателя     | Порт получателя | 
|   :---:            |   :---:          |   :---:           |   :---:         |  
| 220.10.1.86        |    53638         | 77.88.55.66       |    80           |  
{:.mbtablestyle} 

Компьютер из внутренней сети с IP-адресом *220.10.1.86* установил соединение, используя порт *53638*, с Web-сервером *77.88.55.66* (порт *80*).

Для проверки наличия соединения, в таблицу межсетевого экрана добавляется соответствующее поле. 

| IP отправителя     | Порт отправителя | IP получателя     | Порт получателя | Протокол | Флаг   | Соединение | Действие  |
|   :---:            |   :---:          |   :---:           |   :---:         |   :---:  |  :---: | :---:      | :---:     |
| 220.10.1.0/24      |    >1024         | Вне 220.10.1.0/24 |    80           |    TCP   | Любой  | -          | Разрешить | 
| Вне 220.10.1.0/24  |    80            | 220.10.1.0/24     |   >1024         |    TCP   | Ack    | Проверять  | Разрешить | 
| Любой              |    Любой         | Любой             | Любой           |    Любой | Любой  | -          | Запретить | 
{:.mbtablestyle}      

Компьютеры из внутренней сети должны иметь возможность устанавливать соединение с Web-серверами в интернет, поэтому в первом правиле мы не требуем наличия соединения. Но ответы нужно пропускать только от тех Web-серверов, с которыми соединение уже установлено. Поэтому во втором правиле мы проверяем наличие записи о соединении в таблице соединений.

## Другие методы ограничения доступа

Кроме межсетевых экранов могут использоваться также другие методы ограничения доступа, работающие на разных уровнях модели OSI.

На **канальном уровне** возможна фильтрация по MAC-адресам на портах коммутатора. Можно составить список MAC-адресов компьютеров, которым разрешено подключаться к коммутатору. Передавать кадры компьютеров с другими MAC-адресами коммутатор не будет.

На **прикладном уровне** используются *прокси-серверы* (proxy server) и *фильтры содержимого* (content filter). Прокси-серверы работают как межсетевые экраны, но на прикладном уровне: принимают все пакеты, которые передаются по какому-нибудь прикладному протоколу, анализируют заголовки протокола, и могу принять решение, передать пакет дальше, или нет. Часто используются Web-прокси, которые работают по протоколу HTTP. Такие прокси серверы могут ограничивать доступ, например, к социальным сетям с рабочих мест организации. 

Фильтры содержимого также работают на прикладном уровне, но они анализируют не только заголовки пакетов, но и данные. С помощью фильтров содержимого можно, например, заблокировать передачу видео, на каких бы сайтах оно ни размещалось.

Системы *обнаружения вторжений* (intrusion detection system) и *предотвращения вторжений* (intrusion prevention system) работают по принципу, похожему на межсетевые экраны. Отличие заключается в том, что они анализируют не отдельные пакеты, а последовательности пакетов. Они могут определить, например, что злоумышленник подбирает пароль к вашему серверу или ведет сканирование вашей сети. Система обнаружения вторжений предупредит администратора о проходящей атаке, а система предотвращения вторжений попытается автоматически предпринять какие-то действия, чтобы остановить атаку.

## Недостатки межсетевых экранов

Межсетевые экраны обеспечивают безопасность, но нужно быть очень осторожными при их настройке. Ошибка при составлении правил доступа может привести к тому, что все нужные пакеты будет блокироваться межсетевым экраном и сеть будет неработоспособна.

Другая возможная проблема - снижение производительности работы сети при использовании межсетевых экранов. Все пакеты в сети должны быть перехвачены межсетевым экраном и проверены. Если у вас крупная сеть, сложная политика безопасности с большим количеством правил доступа, а межсетевой экран не обладает достаточной производительность, то вся сеть будет работать медленно.

## Итоги

Межсетевые экраны - это устройства или программы, предназначенные для отделения сетей друг от друга. Межсетевые экраны перехватывают все пакеты между сетям и проверяют их на соответствие правилам доступа. При проверке используются IP-адреса отправителей и получателей, порты транспортного уровня, флаги в заголовках сетевых протоколов, а также состояние соединения TCP. Если пакет подходит под разрешающее правило, он передается дальше, в противном случае отбрасывается.






