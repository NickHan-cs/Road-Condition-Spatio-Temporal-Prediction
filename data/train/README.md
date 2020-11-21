# README

## 1. 数据说明

​		该数据是由滴滴平台提供的2019年7月1日至2019年7月30日西安的实时和历史路况信息，以及西安市的道路属性和路网拓扑信息。

## 2. 术语解释

**link**：对完整道路按照拓扑切分后得到的小段，由唯一id标识。出于数据安全考虑，隐去了经纬度等真实地理信息。

**路况状态**：根据道路的平均车速，道路等级等信息对道路通行状态的描述，分为**畅通**，**缓行**，**拥堵**三种状态，分别对应滴滴地图所展示的绿色，黄色，红色。

**时间片**：对时间的离散化描述。一般以**2分钟**为一个单位。2分钟内认为道路的路况状态是统一的。

## 3. 具体数据

### 数据一：历史与实时路况（traffic-fix.tar.gz）

总体格式：

`linkid label current_slice_id future_slice_id;recent_feature;history_feature`

|     字段名称     |                           字段含义                           |
| :--------------: | :----------------------------------------------------------: |
|       link       |                              id                              |
|      label       |               future_slice_id的link的路况状态                |
| current_slice_id |                         当前时间片id                         |
| future_slice_id  |                        待预测时间片id                        |
|  recent_feature  | 当天近期n个时间片路况特征，n=5，时间片之间空格分隔，字段之间,分隔，具体格式：`时间片：路况速度,eta速度,路况状态,参与路况计算的车辆数`。特征都为0时，说明此时间片无车经过 |
| history_feature  | 历史同期在n个时间片路况特征，星期之间;分隔，共4组（-28，-21，-14，-7），每组格式和recent_feature一致（前四周future_slice_id至future_slice_id+n-1时间片时的路况特征） |

* label，在数据中给出了0、1、2、3和4四个离散值，其中1、2、3分别对应于畅通、缓行和拥堵三个标签，4为严重拥堵（在数据处理时可以直接把4变成3）。
* slice_id为负数时代表相对于当天时间前一天的时间，如目前为0点，对应的slice_id为0，则前一天的23点到今天0点的时间片则为-30至0。
* eta速度，猜测应该是系统在当前时间片系下对路况速度的预估。

### 数据二：道路属性（attr.txt）

>  题目官网上没写到path class字段

|  字段名称   | 字段类型  |        字段含义         |
| :---------: | :-------: | :---------------------: |
|   link id   | categoric |        link的id         |
|   length    |  numeric  |  link的长度，以m为单位  |
|  direction  | categoric |     link的通行方向      |
| path class  | categoric |     link的功能等级      |
| speed class | categoric |   link的速度限制等级    |
|   LaneNum   | categoric |      link的车道数       |
| speed limit |  numeric  | link的限速，以m/s为单位 |
|    level    | categoric |       link的level       |
|    width    |  numeric  |  link的宽度，以m为单位  |

### 数据三：路网拓扑（topo.txt）

| key  | value                                         |
| ---- | --------------------------------------------- |
| link | 下游link id1，下游link id2，下游link id3，... |

## 4. 相关统计