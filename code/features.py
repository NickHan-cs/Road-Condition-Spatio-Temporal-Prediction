import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter


def get_his_features(x):
    return [i.split(':')[-1] for i in x.split(' ')]


def get_speed(x):
    speed_list = []
    for i in x:
        speed = i.split(',')[0]
        if float(speed) != 0:
            speed_list.append(speed)
    if not speed_list:
        speed_list.append('0')
    return np.array(speed_list, dtype='float16')


def get_eta_speed(x):
    eta_speed_list = []
    for i in x:
        eta_speed = i.split(',')[1]
        if float(eta_speed) != 0:
            eta_speed_list.append(eta_speed)
    if not eta_speed_list:
        eta_speed_list.append('0')
    return np.array(eta_speed_list, dtype='float16')


def get_car_cnt(x):
    return np.array([i.split(',')[3] for i in x], dtype='int16')


def get_his_label(x):
    his_label_list = []
    for i in x:
        his_label = int(i.split(',')[2])
        his_label = 3 if his_label == 4 else his_label
        if his_label != 0:
            his_label -= 1
            his_label_list.append(his_label)
    if not his_label_list:
        his_label_list.append(0)
    return his_label_list


def generate_features(path, mode="train"):
    traffic_df = pd.read_csv(path, sep=";", header=None)
    traffic_df['link_id'] = traffic_df[0].apply(lambda x: int(x.split(' ')[0]))
    traffic_df = traffic_df.merge(attr_df, on='link_id', how='left')
    if mode == "train":
        traffic_df['future_label'] = traffic_df[0].apply(lambda x: int(x.split(' ')[1]))
        # 当label大于3时，将其视为3(future_label没有为0的数据)
        traffic_df['future_label'] = traffic_df['future_label'].apply(lambda x: 3 if x > 3 else x)
        # 将类别变成0、1、2，便于分类
        traffic_df['future_label'] = traffic_df['future_label'] - 1
        traffic_df['curr_slice_id'] = traffic_df[0].apply(lambda x: int(x.split(' ')[2]))
        traffic_df['future_slice_id'] = traffic_df[0].apply(lambda x: int(x.split(' ')[3]))
    else:
        traffic_df['label'] = -1
        traffic_df['curr_slice_id'] = traffic_df[0].apply(lambda x: int(x.split(' ')[2]))
        traffic_df['future_slice_id'] = traffic_df[0].apply(lambda x: int(x.split(' ')[3]))
    traffic_df['time_difference'] = traffic_df['future_slice_id'] - traffic_df['curr_slice_id']
    traffic_df['curr_features'] = traffic_df[1].apply(lambda x: x.split(' ')[-1].split(':')[-1])
    # 添加强制类型转换
    traffic_df['curr_speed'] = traffic_df['curr_features'].apply(lambda x: float(x.split(',')[0]))
    traffic_df['curr_eta_speed'] = traffic_df['curr_features'].apply(lambda x: float(x.split(',')[1]))
    traffic_df['curr_car_cnt'] = traffic_df['curr_features'].apply(lambda x: int(x.split(',')[3]))
    # 添加新特征，单位长度、单位车道上的车辆数unit_car_cnt
    traffic_df['curr_unit_car_cnt'] = \
        traffic_df.apply(lambda x: x['curr_car_cnt'] / (x['length'] * x['LaneNum']), axis=1)
    traffic_df['curr_label'] = traffic_df['curr_features'].apply(lambda x: int(x.split(',')[2]))
    # 对label进行处理
    traffic_df['curr_label'].apply(lambda x: 3 if x == 4 else x)
    traffic_df['curr_label'] -= 1
    del traffic_df[0], traffic_df['curr_features']

    # tqdm为进度条库
    for i in tqdm(range(1, 6)):
        traffic_df['his_features'] = traffic_df[i].apply(get_his_features)
        time_flag = 'recent' if i == 1 else f'his_{(6 - i) * 7}'
        traffic_df['his_speed'] = traffic_df['his_features'].apply(get_speed)
        traffic_df[f'{time_flag}_speed_min'] = traffic_df['his_speed'].apply(lambda x: x.min())
        traffic_df[f'{time_flag}_speed_max'] = traffic_df['his_speed'].apply(lambda x: x.max())
        traffic_df[f'{time_flag}_speed_mean'] = traffic_df['his_speed'].apply(lambda x: x.mean())
        traffic_df[f'{time_flag}_speed_std'] = traffic_df['his_speed'].apply(lambda x: x.std())

        traffic_df['his_eta_speed'] = traffic_df['his_features'].apply(get_eta_speed)
        traffic_df[f'{time_flag}_eta_speed_min'] = traffic_df['his_eta_speed'].apply(lambda x: x.min())
        traffic_df[f'{time_flag}_eta_speed_max'] = traffic_df['his_eta_speed'].apply(lambda x: x.max())
        traffic_df[f'{time_flag}_eta_speed_mean'] = traffic_df['his_eta_speed'].apply(lambda x: x.mean())
        traffic_df[f'{time_flag}_eta_speed_std'] = traffic_df['his_eta_speed'].apply(lambda x: x.std())

        traffic_df['his_car_cnt'] = traffic_df['his_features'].apply(get_car_cnt)
        traffic_df[f'{time_flag}_car_cnt_min'] = traffic_df['his_car_cnt'].apply(lambda x: x.min())
        traffic_df[f'{time_flag}_car_cnt_max'] = traffic_df['his_car_cnt'].apply(lambda x: x.max())
        traffic_df[f'{time_flag}_car_cnt_mean'] = traffic_df['his_car_cnt'].apply(lambda x: x.mean())
        traffic_df[f'{time_flag}_car_cnt_std'] = traffic_df['his_car_cnt'].apply(lambda x: x.std())
        # 增加新特征unit_car_cnt_mean
        traffic_df[f'{time_flag}_unit_car_cnt_mean'] = traffic_df.apply(
            lambda x: x[f'{time_flag}_car_cnt_mean'] / (x['length'] * x['LaneNum']), axis=1)

        traffic_df['his_label'] = traffic_df['his_features'].apply(get_his_label)
        traffic_df[f'{time_flag}_label'] = traffic_df['his_label'].apply(lambda x: Counter(x).most_common()[0][0])

        traffic_df.drop([i, 'his_features', 'his_speed', 'his_eta_speed', 'his_car_cnt', 'his_label'], axis=1,
                        inplace=True)

    # 若curr时刻的数据出现0值，就用recent时间片相应数据的平均值进行填充
    traffic_df.loc[traffic_df['curr_speed'] == 0, 'curr_speed'] = traffic_df['recent_speed_mean']
    traffic_df.loc[traffic_df['curr_eta_speed'] == 0, 'curr_eta_speed'] = traffic_df['recent_eta_speed_mean']
    traffic_df.loc[traffic_df['curr_label'] == -1, 'curr_label'] = traffic_df['recent_label']

    if mode == 'train':
        traffic_df.to_csv(f"../features/train/{mode}_features_{path.split('/')[-1]}", index=False)
    else:
        traffic_df.to_csv(f"../features/test/test_features_20190801.csv", index=False)


if __name__ == "__main__":
    raw_train_data_path = "../data/train/traffic/20190701.txt"
    raw_test_data_path = "../data/test/20190801_testdata.txt"
    attr_df = pd.read_csv("../data/train/attr.txt", sep='\t',
                          names=['link_id', 'length', 'direction', 'path_class', 'speed_class', 'LaneNum',
                                 'speed_limit', 'level', 'width'], header=None)
    generate_features(raw_train_data_path, mode="train")
    # generate_features(raw_test_data_path, mode="test")
