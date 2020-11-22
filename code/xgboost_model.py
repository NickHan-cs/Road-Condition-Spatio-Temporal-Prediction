import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


def f1_score_eval(labels_pred, valid_df):
    labels_true = valid_df.get_label()
    labels_pred = np.argmax(labels_pred.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels_true, y_pred=labels_pred, average=None)
    scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
    return 'f1_score', scores


def xgb_train(train_df: pd.DataFrame, test_df: pd.DataFrame, used_train_features: list, n_splits: int, split_rs: int,
              is_shuffle=True):
    print('data shape:\ntrain: {}\ntest: {}'.format(train_df.shape, test_df.shape))
    print('Use {} features ...'.format(len(used_train_features)))
    print('Use xgboost to train ...')
    n_class = train_df["future_label"].nunique()
    train_df["label_pred"] = 0
    test_pred = np.zeros((test_df.shape[0], n_class))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["features"] = used_train_features
    # k-交叉验证, random_state是随机数生成使用的随机种子，控制随机状态，设置不同的随机种子，每次构建的模型是不同的
    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=split_rs)
    train_link_id = train_df["link_id"].unique()
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',  # 表示多分类任务，使用softprob函数作为目标函数
        'eval_metric': 'mlogloss',
        'gamma': 0.1,
        'num_class': 3,
        'min_child_weight': 1.1,
        'max_depth': 6,
        'lambda': 5,
        'subsample': 0.7,   # 每个决策树所用的子样本占总样本的比例（作用于样本）
        'colsample_bytree': 0.7,    # 建立树时对特征随机采样的比例（作用于特征）典型值
        'colsample_bylevel': 0.7,
        'eta': 0.05,
        'tree_method': 'exact',
        'seed': 100,
    }
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_link_id), start=1):
        print('the {} training start ...'.format(n_fold))
        train_x, train_y = train_df.loc[train_df["link_id"].isin(train_link_id[train_idx]), used_train_features], \
                           train_df.loc[train_df["link_id"].isin(train_link_id[train_idx]), "future_label"]
        valid_x, valid_y = train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), used_train_features], \
                           train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), "future_label"]
        print(f'for train link:{len(train_idx)}\nfor valid link:{len(valid_idx)}')

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)

        xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=[(dvalid, 'val')],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=f1_score_eval,
            maximize=True
        )

        train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), 'label_pred'] = np.argmax(
            xgb_model.predict(xgb.DMatrix(valid_x), ntree_limit=xgb_model.best_iteration), axis=1)
        print(Counter(np.argmax(xgb_model.predict(xgb.DMatrix(valid_x), ntree_limit=xgb_model.best_iteration), axis=1)))
        test_pred += xgb_model.predict(xgb.DMatrix(test_df[used_train_features]),
                                       ntree_limit=xgb_model.best_iteration) / folds.n_splits

    report = f1_score(train_df["future_label"], train_df["label_pred"], average=None)
    print(classification_report(train_df["future_label"], train_df["label_pred"], digits=4))
    score = report[0] * 0.2 + report[1] * 0.2 + report[2] * 0.6
    print('Score: ', score)
    test_df["label_pred"] = np.argmax(test_pred, axis=1)
    test_df["label"] = np.argmax(test_pred, axis=1) + 1
    '''
    five_folds = [f'fold_{f}_imp' for f in range(1, n_splits + 1)]
    fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    print(fold_importance_df[["features", 'avg_imp']].head(20))
    '''
    return test_df[["link_id", 'curr_slice_id', 'future_slice_id', "label"]], score
