import pandas as pd


def lgb_cb_fusion(label_list) -> int:
    cnt = [0, 0, 0, 0]
    for label in label_list:
        cnt[label] += 1
    if cnt[3] > 0:
        return 3
    if cnt[2] > 0:
        return 2
    return 1


pred_rlt_list = [
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-25-23-39-04_0.5723.csv"),  # 0701
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-25-23-46-34_0.5647.csv"),  # 0702
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-25-12-48-09_0.5686.csv"),  # 0703
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-10-27-16_0.5966.csv"),  # 0704
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-23-53-24_0.5855.csv"),  # 0705
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-18-06-36_0.6077.csv"),  # 0706
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-25-23-53-19_0.5747.csv"),  # 0707
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-19-34-26_0.6033.csv"),  # 0708
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-10-57-40_0.5801.csv"),  # 0709
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-10-44-47_0.5803.csv"),  # 0710
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-21-00-28-23_0.5965.csv"),  # 0711
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-22-26-59_0.5898.csv"),  # 0712
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-09-36-29_0.6163.csv"),  # 0713
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-26-01-43-17_0.5912.csv"),  # 0714
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-11-05-41_0.5907.csv"),  # 0715
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-16-49-26_0.6114.csv"),  # 0716
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-23-33-26_0.5958.csv"),  # 0717
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-18-33-33_0.61.csv"),    # 0718
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-23-17-42_0.6054.csv"),  # 0719
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-09-25-20_0.6273.csv"),  # 0720
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-25-13-01-43_0.5988.csv"),  # 0721
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-17-07-01_0.6296.csv"),  # 0722
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-25-11-44-29_0.5991.csv"),  # 0723
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-25-12-55-40_0.58.csv"),    # 0724
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-20-16-52-04_0.6123.csv"),  # 0725
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-23-42-10_0.6176.csv"),  # 0726
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-11-12-16_0.6178.csv"),  # 0727
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-26-01-49-30_0.5967.csv"),  # 0728
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-11-24-40_0.6022.csv"),  # 0729
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-26-01-55-41_0.5763.csv"),  # 0730

    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-21-54-36_0.5682.csv"),  # 0701
    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-22-50-40_0.564.csv"),   # 0702
    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-22-42-44_0.5653.csv"),  # 0703
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-22-18-57_0.5937.csv"),  # 0704
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-21-17-47_0.5848.csv"),  # 0705
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-14-24-50_0.6085.csv"),  # 0706
    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-18-00-22_0.5717.csv"),  # 0707
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-13-37-20_0.5996.csv"),  # 0708
    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-14-17-26_0.5744.csv"),  # 0709
    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-13-11-36_0.5783.csv"),  # 0710
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-11-18-33_0.5935.csv"),  # 0711
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-22-18-20_0.5863.csv"),  # 0712
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-00-43-04_0.6138.csv"),  # 0713
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-16-02-56_0.5893.csv"),  # 0714
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-15-21-33_0.586.csv"),   # 0715
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-18-42-20_0.6086.csv"),  # 0716
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-01-08-23_0.5937.csv"),  # 0717
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-02-29-13_0.6075.csv"),  # 0718
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-21-27-55_0.6019.csv"),  # 0719
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-01-42-15_0.6263.csv"),  # 0720
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-01-08-10_0.5971.csv"),  # 0721
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-19-37-02_0.626.csv"),   # 0722
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-21-48-36_0.5976.csv"),  # 0723
    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-00-21-32_0.5759.csv"),  # 0724
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-10-02-50_0.6096.csv"),  # 0725
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-01-56-19_0.615.csv"),   # 0726
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-16-20-58_0.6169.csv"),  # 0727
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-23-29-25_0.5923.csv"),  # 0728
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-17-08-15_0.6007.csv"),  # 0729
    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-25-16-49-30_0.5761.csv"),  # 0730

    pd.read_csv("../prediction/LightGBM/20190801_2020-11-20-16-32-06_0.6073.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-21-00-18-23_0.6083.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-17-58-26_0.6009.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-18-21-32_0.6084.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-00-01-54_0.6111.csv"),
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-18-28-16_0.5796.csv"),
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-18-48-28_0.5798.csv"),
    # pd.read_csv("../prediction/LightGBM/20190801_2020-11-24-19-02-21_0.5781.csv"),

    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-20-57-46_0.5989.csv"),
    # pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-07-59-57_0.5822.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-24-09-01-03_0.6096.csv"),
]

for i in range(len(pred_rlt_list)):
    pred_rlt_list[i] = pred_rlt_list[i].rename(columns={'label': f'label{i}'})

df = pd.merge(pred_rlt_list[0], pred_rlt_list[1], on=['link_id', 'curr_slice_id', 'future_slice_id'])
for i in range(2, len(pred_rlt_list)):
    df = pd.merge(df, pred_rlt_list[i], on=['link_id', 'curr_slice_id', 'future_slice_id'])

df["label"] = df.apply(lambda x: lgb_cb_fusion(
    [x[f"label{j}"] for j in range(len(pred_rlt_list))]), axis=1)
for label_i in range(len(pred_rlt_list)):
    del df[f"label{label_i}"]
df = df.rename(columns={'link_id': 'link', 'curr_slice_id': 'current_slice_id'})
df.to_csv("../prediction/model_fusion/lgb_cb_fusion_14.csv", index=False, encoding='utf8')
