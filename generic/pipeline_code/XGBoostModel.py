# XGBoostModel
from .BaseModel import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna
from .CrossValidation import perform_cross_validation

class XGBoostModel(BaseModel):
    def __init__(self, df, target_column, preprocessor=None):
        super().__init__(df, target_column, preprocessor)
        
    def _objective(self, trial, X_train, y_train, random_state, metric='auto'):
        """Define the parameters to optimize for XGBoost"""
        if metric == 'auto':
            metric = 'f1' if self.problem_type == 'classification' else 'mae'
            
        params = {
            # Core Parameters
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            
            # Sampling Parameters (prevent overfitting)
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            
            # Regularization (reduce overfitting)
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 10),
            'gamma': trial.suggest_float('gamma', 0.5, 2.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
            
            # Fixed Parameters
            'random_state': random_state,
            'n_jobs': -1
        }

        model_class = xgb.XGBClassifier if self.problem_type == 'classification' else xgb.XGBRegressor
        
        try:
            cv_score = perform_cross_validation(
                model_class=model_class,
                params=params,
                X_train=X_train,
                y_train=y_train,
                problem_type=self.problem_type,
                metric=metric,
                random_state=random_state,
                trial=trial,
                calculate_metrics=self.calculate_metrics,
                preprocessor=self.preprocessor
            )
            return cv_score
            
        except optuna.TrialPruned:
            raise

    def _create_model(self, params, random_state):
        """Create either an XGBoost classifier or regressor based on problem type"""
        if self.problem_type == 'classification':
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)