import pandas as pd


def lgb_fusion(label_1, label_2, label_3):
    if label_1 == 3 or label_2 == 3 or label_3 == 3:
        return 3
    if label_1 == label_2 or label_2 == label_3:
        return label_2
    return label_3


def lgb_cb_fusion(label_list) -> int:
    cnt = [0, 0, 0, 0]
    for label in label_list:
        cnt[label] += 1
    if cnt[3] > 0:
        return 3
    if cnt[2] > 0:
        return 2
    return 1
    '''
    if cnt[2] > cnt[1]:
        return 2
    if cnt[1] > cnt[2]:
        return 1
    return 2
    '''


pred_rlt_list = [
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-18-06-36_0.6077.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-19-34-26_0.6033.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-21-00-28-23_0.5965.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-09-36-29_0.6163.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-16-49-26_0.6114.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-18-33-33_0.61.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-09-25-20_0.6273.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-23-17-07-01_0.6296.csv"),
    pd.read_csv("../prediction/LightGBM/20190801_2020-11-20-16-52-04_0.6123.csv"),

    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-14-24-50_0.6085.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-13-37-20_0.5996.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-11-18-33_0.5935.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-00-43-04_0.6138.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-18-42-20_0.6086.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-02-29-13_0.6075.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-01-42-15_0.6263.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-23-19-37-02_0.626.csv"),
    pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-10-02-50_0.6096.csv"),
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
df.to_csv("../prediction/model_fusion/lgb_cb_fusion_6.csv", index=False, encoding='utf8')
