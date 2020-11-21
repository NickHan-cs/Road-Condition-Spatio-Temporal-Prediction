import pandas as pd


def fusion(label_1, label_2, label_3):
    if label_1 == 3 or label_2 == 3 or label_3 == 3:
        return 3
    if label_1 == label_2 or label_2 == label_3:
        return label_2
    return label_3


df_1 = pd.read_csv("../prediction/LightGBM/20190801_2020-11-21-00-28-23_0.5965.csv")
df_1 = df_1.rename(columns={'label': 'label1'})
df_2 = pd.read_csv("../prediction/LightGBM/20190801_2020-11-21-00-18-23_0.6083.csv")
df_2 = df_2.rename(columns={'label': 'label2'})
df_3 = pd.read_csv("../prediction/LightGBM/20190801_2020-11-20-16-52-04_0.6123.csv")
df_3 = df_3.rename(columns={'label': 'label3'})
temp_df = pd.merge(df_1, df_2, on=['link_id', 'curr_slice_id', 'future_slice_id'])
df = pd.merge(temp_df, df_3, on=['link_id', 'curr_slice_id', 'future_slice_id'])
df["label"] = df.apply(lambda x: fusion(x.label1, x.label2, x.label3), axis=1)
del df["label1"], df["label2"], df["label3"]
df = df.rename(columns={'link_id': 'link', 'curr_slice_id': 'current_slice_id'})
df.to_csv("../prediction/model_fusion/lgb_model_fusion_1.csv", index=False, encoding='utf8')
