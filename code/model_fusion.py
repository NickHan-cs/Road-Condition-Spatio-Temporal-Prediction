import pandas as pd


def lgb_fusion(label_1, label_2, label_3):
    if label_1 == 3 or label_2 == 3 or label_3 == 3:
        return 3
    if label_1 == label_2 or label_2 == label_3:
        return label_2
    return label_3


def lgb_cb_fusion(label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9, label_10):
    cnt = [0, 0, 0, 0]
    cnt[label_1] += 1
    cnt[label_2] += 1
    cnt[label_3] += 1
    cnt[label_4] += 1
    cnt[label_5] += 1
    cnt[label_6] += 1
    cnt[label_7] += 1
    cnt[label_8] += 1
    cnt[label_9] += 1
    cnt[label_10] += 1
    if cnt[3] > 0:
        return 3
    if cnt[2] > cnt[1]:
        return 2
    if cnt[1] > cnt[2]:
        return 1
    return label_5


df_1 = pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-18-06-36_0.6077.csv")
df_1 = df_1.rename(columns={'label': 'label1'})
df_2 = pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-19-34-26_0.6033.csv")
df_2 = df_2.rename(columns={'label': 'label2'})
df_3 = pd.read_csv("../prediction/LightGBM/20190801_2020-11-21-00-28-23_0.5965.csv")
df_3 = df_3.rename(columns={'label': 'label3'})
df_4 = pd.read_csv("../prediction/LightGBM/20190801_2020-11-22-18-21-32_0.6084.csv")
df_4 = df_4.rename(columns={'label': 'label4'})
df_5 = pd.read_csv("../prediction/LightGBM/20190801_2020-11-20-16-52-04_0.6123.csv")
df_5 = df_5.rename(columns={'label': 'label5'})

df_6 = pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-14-24-50_0.6085.csv")
df_6 = df_6.rename(columns={'label': 'label6'})
df_7 = pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-13-37-20_0.5996.csv")
df_7 = df_7.rename(columns={'label': 'label7'})
df_8 = pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-11-18-33_0.5935.csv")
df_8 = df_8.rename(columns={'label': 'label8'})
df_9 = pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-02-29-13_0.6075.csv")
df_9 = df_9.rename(columns={'label': 'label9'})
df_10 = pd.read_csv("../prediction/CatBoost/20190801_2020-11-22-10-02-50_0.6096.csv")
df_10 = df_10.rename(columns={'label': 'label10'})

temp_df_1 = pd.merge(df_1, df_2, on=['link_id', 'curr_slice_id', 'future_slice_id'])
temp_df_2 = pd.merge(temp_df_1, df_3, on=['link_id', 'curr_slice_id', 'future_slice_id'])
temp_df_3 = pd.merge(temp_df_2, df_4, on=['link_id', 'curr_slice_id', 'future_slice_id'])
temp_df_4 = pd.merge(temp_df_3, df_5, on=['link_id', 'curr_slice_id', 'future_slice_id'])
temp_df_5 = pd.merge(temp_df_4, df_6, on=['link_id', 'curr_slice_id', 'future_slice_id'])
temp_df_6 = pd.merge(temp_df_5, df_7, on=['link_id', 'curr_slice_id', 'future_slice_id'])
temp_df_7 = pd.merge(temp_df_6, df_8, on=['link_id', 'curr_slice_id', 'future_slice_id'])
temp_df_8 = pd.merge(temp_df_7, df_9, on=['link_id', 'curr_slice_id', 'future_slice_id'])
df = pd.merge(temp_df_8, df_10, on=['link_id', 'curr_slice_id', 'future_slice_id'])
df["label"] = df.apply(lambda x: lgb_cb_fusion(x.label1, x.label2, x.label3, x.label4, x.label5, x.label6, x.label7,
                                               x.label8, x.label9, x.label10), axis=1)
del df["label1"], df["label2"], df["label3"], df["label4"], df["label5"], df["label6"], df["label7"], df["label8"], \
    df["label9"], df["label10"]
df = df.rename(columns={'link_id': 'link', 'curr_slice_id': 'current_slice_id'})
df.to_csv("../prediction/model_fusion/lgb_cb_fusion_3.csv", index=False, encoding='utf8')
