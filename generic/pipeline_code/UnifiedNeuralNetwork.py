# UnifiedNeuralNetwork
from .BaseModel import BaseModel
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np
import pandas as pd
import optuna
from .CrossValidation import perform_cross_validation

class UnifiedNeuralNetwork(BaseModel):
    def __init__(self, df, target_column, preprocessor=None):
        super().__init__(df, target_column, preprocessor)
        
    def _objective(self, trial, X_train, y_train, random_state, metric='auto'):
        """Define the parameters to optimize for Neural Network with pruning support"""
        if metric == 'auto':
            metric = 'f1' if self.problem_type == 'classification' else 'mae'
            
        # Get parameters based on the task
        params = self._get_params_for_task(trial, random_state)
        
        # Get the right model class based on problem type
        model_class = MLPClassifier if self.problem_type == 'classification' else MLPRegressor
        
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
                preprocessor=self.preprocessor
            )
            return cv_score
            
        except optuna.TrialPruned:
            # Re-raise pruning exception for Optuna
            raise
    
    def _get_params_for_task(self, trial, random_state):
        """Generate parameters based on the task type"""
        if self.problem_type == 'classification':
            return {
                'hidden_layer_sizes': (trial.suggest_int('hidden_layer_sizes', 5, 20),),  # Neurons
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.01, log=True),  # Learning rate
                'max_iter': trial.suggest_int('max_iter', 20, 100),  # Max iterations
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),  # L2 penalty
                'batch_size': trial.suggest_int('batch_size', 32, 64),  # Batch size
                'activation': 'relu',  # Activation function
                'early_stopping': True,  # Stop early
                'validation_fraction': 0.1,  # Validation split
                'n_iter_no_change': 5,  # Patience
                'solver': 'adam',  # Optimizer
                'random_state': random_state
            }
        else:  # Regression
            return {
                'hidden_layer_sizes': (trial.suggest_int('hidden_layer_sizes', 10, 30),),  # Neurons
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.01, log=True),  # Learning rate
                'max_iter': trial.suggest_int('max_iter', 50, 200),  # Max iterations
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),  # L2 penalty
                'batch_size': trial.suggest_int('batch_size', 32, 64),  # Batch size
                'activation': 'relu',  # Activation function
                'early_stopping': True,  # Stop early
                'validation_fraction': 0.1,  # Validation split
                'n_iter_no_change': 5,  # Patience
                'solver': 'adam',  # Optimizer
                'random_state': random_state
            }

    def _create_model(self, params, random_state):
        """Create either a NN classifier or regressor based on problem type"""
        if self.problem_type == 'classification':
            return MLPClassifier(**params)
        return MLPRegressor(**params)