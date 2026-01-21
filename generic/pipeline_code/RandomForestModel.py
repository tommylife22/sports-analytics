# RandomForestModel
from .BaseModel import BaseModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np
import optuna
from .CrossValidation import perform_cross_validation

class RandomForestModel(BaseModel):
    def __init__(self, df, target_column, preprocessor=None):
        super().__init__(df, target_column, preprocessor)
        
    def _objective(self, trial, X_train, y_train, random_state, metric='auto'):
        """Define the parameters to optimize for Random Forest"""
        if metric == 'auto':
            metric = 'f1' if self.problem_type == 'classification' else 'mae'
            
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 30, 120, step=10),
            'max_depth': trial.suggest_int('max_depth', 2, 5, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 4, 8),
            'max_features': trial.suggest_float('max_features', 0.4, 0.8),
            'random_state': random_state,
            'n_jobs': -1  # Use all CPU cores
        }
        
        # Get the right model class based on problem type
        model_class = RandomForestClassifier if self.problem_type == 'classification' else RandomForestRegressor
        
        # Use our common cross-validation module
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
                preprocessor=self.preprocessor  # Pass preprocessor
            )
            return cv_score
            
        except optuna.TrialPruned:
            # Re-raise pruning exception for Optuna
            raise
    
    def _create_model(self, params, random_state):
        """Create either a RF classifier or regressor based on problem type"""
        if self.problem_type == 'classification':
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)