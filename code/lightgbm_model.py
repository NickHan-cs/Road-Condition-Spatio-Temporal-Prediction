import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score
import warnings

warnings.filterwarnings("ignore")


def f1_score_eval(labels_pred, valid_df):
    labels_true = valid_df.get_label()
    labels_pred = np.argmax(labels_pred.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels_true, y_pred=labels_pred, average=None)
    scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
    return 'f1_score', scores, True


def lgb_train(train_df: pd.DataFrame, test_df: pd.DataFrame, used_train_features: list, n_splits: int, split_rs: int,
              is_shuffle=True, use_cart=False, cate_cols=None):
    # cate_cols是做什么的？
    if not cate_cols:
        cate_cols = []
    print('data shape:\ntrain: {}\ntest: {}'.format(train_df.shape, test_df.shape))
    print('Use {} features ...'.format(len(used_train_features)))
    print('Use lightgbm to train ...')
    n_class = train_df["future_label"].nunique()
    train_df["label_pred"] = 0
    test_pred = np.zeros((test_df.shape[0], n_class))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["features"] = used_train_features
    # k-交叉验证, random_state是随机数生成使用的随机种子，控制随机状态，设置不同的随机种子，每次构建的模型是不同的
    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=split_rs)
    train_link_id = train_df["link_id"].unique()
    params = {
        'learning_rate': 0.05,  # 学习率
        'boosting_type': 'gbdt',  # 基学习器模型算法，'gbdt'--传统的梯度提升决策树
        'objective': 'multiclass',  # 表示多分类任务，使用softmax函数作为目标函数
        'metric': 'None',
        'num_leaves': 31,  # 叶子节点数，默认31
        'num_class': n_class,
        'feature_fraction': 0.8,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 每k次执行bagging
        'seed': 1,
        'bagging_seed': 12,  # 表示bagging的随机数种子
        'feature_fraction_seed': 24,  # 表示feature_fraction的随机数种子
        'min_data_in_leaf': 20,  # 表示一个叶子节点上包含的最少样本数量，默认20
        'nthread': -1,
        'verbose': -1
    }

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_link_id), start=1):
        print('the {} training start ...'.format(n_fold))
        train_x, train_y = train_df.loc[train_df["link_id"].isin(train_link_id[train_idx]), used_train_features], \
                           train_df.loc[train_df["link_id"].isin(train_link_id[train_idx]), "future_label"]
        valid_x, valid_y = train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), used_train_features], \
                           train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), "future_label"]
        print(f'for train link:{len(train_idx)}\nfor valid link:{len(valid_idx)}')

        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_features=cate_cols)
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_features=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        lgb_model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=f1_score_eval
        )
        fold_importance_df[f'fold_{n_fold}_imp'] = lgb_model.feature_importance(importance_type='gain')
        train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), 'label_pred'] = np.argmax(
            lgb_model.predict(valid_x, num_iteration=lgb_model.best_iteration), axis=1)
        test_pred += lgb_model.predict(test_df[used_train_features],
                                       num_iteration=lgb_model.best_iteration) / folds.n_splits

    report = f1_score(train_df["future_label"], train_df["label_pred"], average=None)
    print(classification_report(train_df["future_label"], train_df["label_pred"], digits=4))
    score = report[0] * 0.2 + report[1] * 0.2 + report[2] * 0.6
    print('Score: ', score)
    test_df["label_pred"] = np.argmax(test_pred, axis=1)
    test_df["label"] = np.argmax(test_pred, axis=1) + 1
    five_folds = [f'fold_{f}_imp' for f in range(1, n_splits + 1)]
    fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    print(fold_importance_df[["features", 'avg_imp']].head(20))
    return test_df[["link_id", 'curr_slice_id', 'future_slice_id', "label"]], score
