# CrossValidation

# A reusable module for performing cross-validation across all model types
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
from sklearn.exceptions import ConvergenceWarning

def perform_cross_validation(
    model_class, 
    params, 
    X_train, 
    y_train, 
    problem_type,
    metric='auto',
    random_state=42,
    n_splits=5,
    trial=None,
    calculate_metrics=None,
    preprocessor=None,
    cv_type="random"
):
    """
    Perform cross-validation for any model type with optional Optuna pruning.
    
    Parameters:
    -----------
    model_class : class
        The model class to instantiate (e.g., XGBRegressor, RandomForestClassifier)
    params : dict
        Parameters to pass to the model constructor
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    problem_type : str
        Either 'classification' or 'regression'
    metric : str
        Metric to optimize (e.g., 'mae', 'accuracy', 'r2')
    random_state : int
        Random seed for reproducibility
    n_splits : int
        Number of folds for cross-validation
    trial : optuna.trial.Trial, optional
        Optuna trial object for pruning
    calculate_metrics : function, optional
        Function to calculate metrics (if None, uses the default implementation)
    preprocessor : DataPreprocessor, optional
        Preprocessor object that contains scaling information
        
    Returns:
    --------
    float
        Mean score across all folds
    """
    # Default metric if none specified
    if metric == 'auto':
        metric = 'accuracy' if problem_type == 'classification' else 'mae'
    
    # Initialize cross-validation
    if cv_type == "time":
        # Assumes X_train is already sorted chronologically
        splitter = TimeSeriesSplit(n_splits=n_splits)
    else:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    
    # Track scores for each fold
    scores = []
    
    # Define a default metric calculation function if none provided
    if calculate_metrics is None:
        def default_calculate_metrics(y_true, y_pred, problem_type):
            if problem_type == 'classification':
                return {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                }
            else:
                return {
                    'r2': r2_score(y_true, y_pred),
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred)
                }
        
        metric_function = default_calculate_metrics
    else:
        # If using BaseModel's calculate_metrics, we need to handle prefixes
        def adapter_calculate_metrics(y_true, y_pred, problem_type):
            # Call the provided calculate_metrics function with an empty prefix
            metrics = calculate_metrics(y_true, y_pred, '')
            
            # Create a clean version of metrics without prefixes
            clean_metrics = {}
            for key, value in metrics.items():
                # Remove any prefix from the key
                clean_key = key.split('_')[-1] if '_' in key else key
                clean_metrics[clean_key] = value
                
            return clean_metrics
            
        metric_function = adapter_calculate_metrics
    
    # Run cross-validation
    for step, (train_idx, val_idx) in enumerate(splitter.split(X_train)): 
        # enumerate adds a counter (step) to track which fold we're processing (0, 1, 2, ...)
        # Split data for this fold
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # If preprocessor exists, fit and transform for this fold
        if preprocessor and hasattr(preprocessor, 'fit_scaler'):
            # Fit scaler on fold training data
            preprocessor.fit_scaler(X_fold_train)
            
            # Transform both fold sets
            X_fold_train = preprocessor.transform_data(X_fold_train)
            X_fold_val = preprocessor.transform_data(X_fold_val)
        
        # Create and train the model
        model = model_class(**params)
        
        # Suppress convergence warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            # Different fitting behavior for XGBoost
            if 'XGB' in model.__class__.__name__:
                model.fit(
                    X_fold_train, 
                    y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False
                )
            else:
                model.fit(X_fold_train, y_fold_train)
        
        # Generate predictions
        y_pred = model.predict(X_fold_val)
        
        # Calculate metrics and get the one we're optimizing for
        fold_metrics = metric_function(y_fold_val, y_pred, problem_type)
        
        # Verify the metric exists in the results
        if metric not in fold_metrics:
            available_metrics = list(fold_metrics.keys())
            raise KeyError(f"Metric '{metric}' not found in available metrics: {available_metrics}")
            
        fold_score = fold_metrics[metric]
        scores.append(fold_score)
        
        # Report to Optuna for pruning if trial is provided
        if trial is not None:
            mean_score = np.mean(scores)
            trial.report(mean_score, step)
            
            if trial.should_prune():
                print(f"🔴 PRUNING Trial #{trial.number}")
                raise optuna.TrialPruned()
    
    # Return the mean score across all folds
    final_score = np.mean(scores)
    if trial is not None:
        print(f"✅ Trial #{trial.number} completed")
    return final_score

def get_scorer(metric, problem_type):
    """
    Get the appropriate sklearn scorer name.
    
    Parameters:
    -----------
    metric : str
        The metric to optimize (e.g., 'mae', 'accuracy')
    problem_type : str
        Either 'classification' or 'regression'
        
    Returns:
    --------
    str
        The corresponding scorer name for cross_val_score
    """
    if problem_type == 'classification':
        return {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        }.get(metric, 'accuracy')
    else:
        return {
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'r2': 'r2'
        }.get(metric, 'neg_mean_absolute_error')