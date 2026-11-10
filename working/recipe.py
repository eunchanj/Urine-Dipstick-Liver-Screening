"""Training recipe of the calibrated ensemble (no data access; used by s02).

Each learner is tuned by Optuna (TPE sampler, seed 42) on the mean AUROC of 3 shuffled stratified folds. The three
tuned learners are combined by soft voting and calibrated by isotonic regression over 5 folds, so the fitted model
predicts the mean of 5 calibrated fold ensembles.
"""
import warnings

import numpy as np
# scikit-learn is imported before LightGBM on purpose: in the other order XGBoost can hang (two OpenMP
# runtimes in one process).
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna

from runlog import say

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

LEARNERS = ("lgb", "xgb", "cat")
TUNING_FOLDS, SEED, SAMPLER = 3, 42, "TPE"
VOTING, CALIBRATION, CALIBRATION_FOLDS, CLASS_WEIGHTING = "soft", "isotonic", 5, "none"


def tune_lightgbm(X, y, n_trials):
    """Optuna search for LightGBM; returns (best parameters, best mean AUROC, study)."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.10, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            model = lgb.LGBMClassifier(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            y_prob = model.predict_proba(X.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va_idx], y_prob))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study


def tune_xgboost(X, y, n_trials):
    """Optuna search for XGBoost; returns (best parameters, best mean AUROC, study)."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.10, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            y_prob = model.predict_proba(X.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va_idx], y_prob))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study


def tune_catboost(X, y, n_trials):
    """Optuna search for CatBoost; returns (best parameters, best mean AUROC, study)."""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300, step=50),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.10, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1,
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            model = cb.CatBoostClassifier(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            y_prob = model.predict_proba(X.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va_idx], y_prob))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study


def build_tuned_voting_ensemble(X_train, y_train, n_trials):
    """Tune the three learners, then fit the soft-voting ensemble with isotonic calibration; returns (model, info,
    studies)."""
    say(f"    Tuning LightGBM ({n_trials} trials)...")
    lgb_params, lgb_auc, lgb_study = tune_lightgbm(X_train, y_train, n_trials)

    say(f"    Tuning XGBoost ({n_trials} trials)...")
    xgb_params, xgb_auc, xgb_study = tune_xgboost(X_train, y_train, n_trials)

    say(f"    Tuning CatBoost ({n_trials} trials)...")
    cb_params, cb_auc, cb_study = tune_catboost(X_train, y_train, n_trials)

    lgb_params.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    xgb_params.update({'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
                        'use_label_encoder': False, 'eval_metric': 'logloss'})
    cb_params.update({'random_seed': 42, 'verbose': False, 'thread_count': -1})

    estimators = [
        ('lgb', lgb.LGBMClassifier(**lgb_params)),
        ('xgb', xgb.XGBClassifier(**xgb_params)),
        ('cat', cb.CatBoostClassifier(**cb_params)),
    ]
    voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)

    say("    Fitting calibrated voting ensemble (isotonic, 5-fold CV)...")
    cal_model = CalibratedClassifierCV(voting, method='isotonic', cv=5, n_jobs=1)
    cal_model.fit(X_train, y_train)

    return cal_model, {'lgb': lgb_params, 'xgb': xgb_params, 'cat': cb_params,
                       'cv_aucs': {'lgb': lgb_auc, 'xgb': xgb_auc, 'cat': cb_auc}}, \
        {'lgb': lgb_study, 'xgb': xgb_study, 'cat': cb_study}
