import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


class WeightedF1Metric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight=None):
        global cnt
        assert(len(approxes) == 3)
        labels_true = np.array(target)
        labels_pred = np.argmax(approxes, axis=0)
        scores = f1_score(y_true=labels_true, y_pred=labels_pred, average=None)
        scores = scores[0] * 0.2 + scores[1] * 0.2 + scores[2] * 0.6
        return scores, 1


def cb_train(train_df: pd.DataFrame, test_df: pd.DataFrame, used_train_features: list, n_splits: int, split_rs: int,
              is_shuffle=True):
    print('data shape:\ntrain: {}\ntest: {}'.format(train_df.shape, test_df.shape))
    print('Use {} features ...'.format(len(used_train_features)))
    print('Use catboost to train ...')
    n_class = train_df["future_label"].nunique()
    train_df["label_pred"] = 0
    test_pred = np.zeros((test_df.shape[0], n_class))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["features"] = used_train_features
    # k-交叉验证, random_state是随机数生成使用的随机种子，控制随机状态，设置不同的随机种子，每次构建的模型是不同的
    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=split_rs)
    train_link_id = train_df["link_id"].unique()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_link_id), start=1):
        print('the {} training start ...'.format(n_fold))
        train_x, train_y = train_df.loc[train_df["link_id"].isin(train_link_id[train_idx]), used_train_features], \
                           train_df.loc[train_df["link_id"].isin(train_link_id[train_idx]), "future_label"]
        valid_x, valid_y = train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), used_train_features], \
                           train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), "future_label"]
        print(f'for train link:{len(train_idx)}\nfor valid link:{len(valid_idx)}')
        cb_model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=5,
            random_seed=888,
            early_stopping_rounds=200,
            verbose=100,
            loss_function='MultiClass',
            eval_metric=WeightedF1Metric(),
            use_best_model=True,
        )
        cb_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)])
        fold_importance_df[f'fold_{n_fold}_imp'] = cb_model.get_feature_importance()
        train_df.loc[train_df["link_id"].isin(train_link_id[valid_idx]), 'label_pred'] = np.argmax(
            cb_model.predict_proba(valid_x, ntree_end=cb_model.best_iteration_), axis=1)
        test_pred += cb_model.predict_proba(test_df[used_train_features], ntree_end=cb_model.best_iteration_) / folds.n_splits

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
