"""
XGBoost classifier and cross-validated hyperparameter search for the
clinical HCC risk model.
"""

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

DEFAULT_PARAMS = dict(
    objective="binary:logistic",
    eval_metric=["logloss", "auc"],
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

# Grid searched with 5-fold CV on the training split.
PARAM_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 300, 500],
}


def create_model(**overrides):
    params = {**DEFAULT_PARAMS, **overrides}
    model = XGBClassifier(**params)
    print(f"[INFO] XGBClassifier created — n_estimators={params['n_estimators']}, "
          f"max_depth={params['max_depth']}, learning_rate={params['learning_rate']}")
    return model


def cross_validated_search(X_train, y_train, seed=42, n_splits=5, param_grid=None):
    """
    Grid search with StratifiedKFold on the training split only. The final
    model is re-fit afterwards with early stopping on the validation split.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base_model = create_model()
    search = GridSearchCV(
        base_model,
        param_grid=param_grid or PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X_train, y_train)
    print(f"[INFO] CV search best params: {search.best_params_} "
          f"(mean AUC={search.best_score_:.4f})")
    return search
