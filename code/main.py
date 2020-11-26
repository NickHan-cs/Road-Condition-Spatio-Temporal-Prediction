import pandas as pd
from lightgbm_model import lgb_train
from catboost_model import cb_train
from xgboost_model import xgb_train
import time


if __name__ == "__main__":
    train_data_df = pd.read_csv("../features/train/train_features_20190730.txt")
    test_data_df = pd.read_csv("../features/test/test_features_20190801.csv")
    not_used_columns = [
        'link_id', 'future_label', 'curr_slice_id', 'label_pred',

        # 'future_slice_id',

        # 'time_difference',

        # 'recent_speed_min', 'recent_speed_max', 'recent_speed_std',
        # 'recent_eta_speed_min', 'recent_eta_speed_max', 'recent_eta_speed_std',
        # 'recent_car_cnt_min', 'recent_car_cnt_max', 'recent_car_cnt_std',

        # 'his_28_speed_min', 'his_28_speed_max', 'his_28_speed_std',
        # 'his_28_eta_speed_min', 'his_28_eta_speed_max', 'his_28_eta_speed_std',
        # 'his_28_car_cnt_min', 'his_28_car_cnt_max', 'his_28_car_cnt_std',

        # 'his_21_speed_min', 'his_21_speed_max', 'his_21_speed_std',
        # 'his_21_eta_speed_min', 'his_21_eta_speed_max', 'his_21_eta_speed_std',
        # 'his_21_car_cnt_min', 'his_21_car_cnt_max', 'his_21_car_cnt_std',

        # 'his_14_speed_min', 'his_14_speed_max', 'his_14_speed_std',
        # 'his_14_eta_speed_min', 'his_14_eta_speed_max', 'his_14_eta_speed_std',
        # 'his_14_car_cnt_min', 'his_14_car_cnt_max', 'his_14_car_cnt_std',

        # 'his_7_speed_min', 'his_7_speed_max', 'his_7_speed_std',
        # 'his_7_eta_speed_min', 'his_7_eta_speed_max', 'his_7_eta_speed_std',
        # 'his_7_car_cnt_min', 'his_7_car_cnt_max', 'his_7_car_cnt_std',

        # 'curr_car_cnt', 'recent_car_cnt_mean', 'his_28_car_cnt_mean', 'his_21_car_cnt_mean', 'his_14_car_cnt_mean', 'his_7_car_cnt_mean',

        # 'curr_eta_speed', 'recent_eta_speed_mean', 'his_28_eta_speed_mean', 'his_21_eta_speed_mean', 'his_14_eta_speed_mean', 'his_7_eta_speed_mean',

        # 'curr_unit_car_cnt', 'his_28_unit_car_cnt_mean', 'his_21_unit_car_cnt_mean', 'his_14_unit_car_cnt_mean', 'his_7_unit_car_cnt_mean',
    ]
    used_columns = [i for i in train_data_df if i not in not_used_columns]

    result_file, result_score = lgb_train(train_data_df, test_data_df, used_columns, 5, 666)
    result_score = round(result_score, 4)
    now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    result_file_path = f"../prediction/LightGBM/20190801_{now}_{result_score}.csv"
    result_file.to_csv(result_file_path, index=False, encoding='utf8')
