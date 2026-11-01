"""Train the final model and save it to model.bin.

Same setup as the best model in notebook.ipynb: XGBoost on all features,
parameters picked by 5-fold cross-validation grouped by player.
First fits on sessions before TEST_FROM and reports AUC on the newest
game version, then refits on all data for the saved model.

Usage: python train.py [waves_csv] [output_bin]
"""
import pickle
import sys

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from features import TARGET, add_features, feature_columns, to_dicts

TEST_FROM = '2026-08-16'  # game version 1.6 and later
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, max_depth=6, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, n_jobs=8, random_state=1, verbosity=0,
)


def fit(df, categorical, numerical):
    pipeline = make_pipeline(DictVectorizer(sparse=False), xgb.XGBClassifier(**XGB_PARAMS))
    return pipeline.fit(to_dicts(df, categorical, numerical), df[TARGET])


def main(data_path, output_path):
    df = add_features(pd.read_csv(data_path))
    categorical, numerical = feature_columns(df)

    is_test = df.started_at.str[:10] >= TEST_FROM
    df_train, df_test = df[~is_test], df[is_test]
    pipeline = fit(df_train, categorical, numerical)
    y_pred = pipeline.predict_proba(to_dicts(df_test, categorical, numerical))[:, 1]
    print(f'train {len(df_train)} waves, test {len(df_test)} waves (from {TEST_FROM})')
    print(f'test AUC: {roc_auc_score(df_test[TARGET], y_pred):.3f}')

    pipeline = fit(df, categorical, numerical)
    with open(output_path, 'wb') as f:
        pickle.dump({'pipeline': pipeline, 'categorical': categorical, 'numerical': numerical}, f)
    print(f'final model on {len(df)} waves -> {output_path}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/waves.csv',
         sys.argv[2] if len(sys.argv) > 2 else 'model.bin')
