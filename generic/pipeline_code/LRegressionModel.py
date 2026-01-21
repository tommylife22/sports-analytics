# LRegressionModel
from .BaseModel import BaseModel
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np
import optuna
from .CrossValidation import perform_cross_validation

class LRegressionModel(BaseModel):
    def __init__(self, df, target_column, preprocessor=None):
        super().__init__(df, target_column, preprocessor)
        
    def _objective(self, trial, X_train, y_train, random_state, metric='auto'):
        """Find the best settings for Linear/Logistic Regression with pruning support"""
        if metric == 'auto':
            metric = 'f1' if self.problem_type == 'classification' else 'mae'
            
        # Different parameters based on problem type
        if self.problem_type == 'classification':
            # First, let's try liblinear which often converges better
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
            
            params = {
                'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                'solver': solver,
                'random_state': random_state
            }
            
            # Different max_iter for different solvers
            if solver == 'lbfgs':
                params['max_iter'] = trial.suggest_int('max_iter', 1000, 5000)
            elif solver == 'saga':
                params['max_iter'] = trial.suggest_int('max_iter', 1000, 5000)
                params['penalty'] = 'l2'  # saga requires penalty
            else:  # liblinear
                params['max_iter'] = trial.suggest_int('max_iter', 500, 2000)
                
            model_class = LogisticRegression
        else:
            params = {}
            model_class = LinearRegression
        
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
            raise
    
    def _create_model(self, params, random_state):
        """Create either a linear or logistic regression model based on problem type"""
        if self.problem_type == 'classification':
            return LogisticRegression(**params)
        else:
            return LinearRegression()