# BaseModel

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,  # For Classification
    mean_squared_error, r2_score, mean_absolute_error        # For Regression
)
import plotly.io as pio
import optuna

import sys
import os

# Add project root to path for generic imports (Jupyter notebook version)
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.pipeline_code.CrossValidation import perform_cross_validation, get_scorer
from generic.pipeline_code.FeatureImportance import calculate_feature_importance
import warnings
from sklearn.exceptions import ConvergenceWarning

class BaseModel:
    """
    A base class for all machine learning models.
    """
    
    def __init__(self, df, target_variable, preprocessor=None):
        """Start up our model with data and what we want to predict"""
        self.df = df
        self.target_variable = target_variable
        self.preprocessor = preprocessor
        self.model = None
        self.problem_type = None
        self.study = None
        self.best_params = None
        self.feature_importance = None
        self.explanation = None
        self.train_metrics = None
        self.test_metrics = None
        self.test_size = None
        self.model_type = self.__class__.__name__.replace('Model', '').lower()
        
    def _get_cv_score(self, model, X_train, y_train, metric='auto'):
        """
        Perform cross-validation and return mean score.
        This is a common function used by all models.
        """
        # Use cross-validation for more robust performance estimate
        scoring = get_scorer(metric, self.problem_type)
        
        try:
            cv_scores = cross_val_score(
                model, 
                X_train, 
                y_train, 
                cv=5,
                scoring=scoring
            )
            
            # For metrics that return negative values (MAE, MSE)
            if scoring.startswith('neg_'):
                return -cv_scores.mean()  # Convert back to positive for minimization
            return cv_scores.mean()
            
        except Exception as e:
            print(f"CV Error: {e}")
            return float('-inf') if self.study.direction == 'maximize' else float('inf')
    
    def _get_scoring_metric(self, metric='auto'):
        """Get the appropriate scoring metric for cross_val_score"""
        return get_scorer(metric, self.problem_type)
    
    def detect_problem_type(self):
        """Figure out if we're doing classification or regression"""
        unique_values = np.unique(self.df[self.target_variable])
        
        # If we have less than 10 unique values, probably classification
        if len(unique_values) < 10 or (
            all(isinstance(val, (int, np.integer)) for val in unique_values) and 
            len(unique_values) < 10
        ):
            return 'classification'
        return 'regression'
    
    def _time_based_split(self, X, y, test_size=0.2):
        """
        Assumes X is already sorted by time.
        """
        n = len(X)
        split_idx = int(n * (1.0 - test_size))

        X_train = X.iloc[:split_idx].reset_index(drop=True)
        X_test  = X.iloc[split_idx:].reset_index(drop=True)

        y_train = y.iloc[:split_idx].reset_index(drop=True)
        y_test  = y.iloc[split_idx:].reset_index(drop=True)

        return X_train, X_test, y_train, y_test, split_idx

    def split_data(
        self,
        random_state=42,
        test_size=0.2,
        split_type="random",
        time_col=None
    ):
        df = self.df.copy()

        # -----------------------------
        # Define columns to drop ALWAYS
        # -----------------------------
        drop_cols = [self.target_variable]
        if time_col is not None:
            drop_cols.append(time_col)

        # -----------------------------
        # TIME-BASED SPLIT
        # -----------------------------
        if split_type == "time":
            if time_col is None:
                raise ValueError("time_col must be provided for time-based split")

            # Ensure sorted order
            df = df.sort_values(time_col).reset_index(drop=True)

            y = df[self.target_variable]
            X = df.drop(columns=drop_cols)

            split_idx = int(len(X) * (1.0 - test_size))

            X_train = X.iloc[:split_idx].reset_index(drop=True)
            X_test  = X.iloc[split_idx:].reset_index(drop=True)

            y_train = y.iloc[:split_idx].reset_index(drop=True)
            y_test  = y.iloc[split_idx:].reset_index(drop=True)

            # ---- HARD SAFETY CHECKS ----
            assert time_col not in X_train.columns
            assert time_col not in X_test.columns

            train_max_date = df[time_col].iloc[:split_idx].max()
            test_min_date  = df[time_col].iloc[split_idx:].min()

            assert train_max_date <= test_min_date, (
                f"Time leakage detected: "
                f"train_max_date={train_max_date}, "
                f"test_min_date={test_min_date}"
            )

        # -----------------------------
        # RANDOM SPLIT
        # -----------------------------
        else:
            y = df[self.target_variable]
            X = df.drop(columns=drop_cols)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if self.problem_type == "classification" else None
            )

            assert time_col not in X_train.columns
            assert time_col not in X_test.columns

        # -----------------------------
        # SCALING (train-only)
        # -----------------------------
        if self.preprocessor and hasattr(self.preprocessor, "fit_scaler"):
            self.preprocessor.fit_scaler(X_train)
            X_train = self.preprocessor.transform_data(X_train)
            X_test  = self.preprocessor.transform_data(X_test)

        return X_train, X_test, y_train, y_test


    
    def calculate_metrics(self, y_true, y_pred, prefix=''):
        if self.problem_type == 'classification':
            return {
                f'{prefix}accuracy': accuracy_score(y_true, y_pred),
                f'{prefix}precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                f'{prefix}recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                f'{prefix}f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                f'{prefix}auc': roc_auc_score(y_true, y_pred),
            }
        else:
            return {
                f'{prefix}r2': r2_score(y_true, y_pred),
                f'{prefix}mse': mean_squared_error(y_true, y_pred),
                f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                f'{prefix}mae': mean_absolute_error(y_true, y_pred)
            }
        
    def _objective(self, trial, X_train, y_train, random_state, metric='auto'):
        """Base objective function for optimization with pruning support"""
        # Get metric to optimize
        if metric == 'auto':
            metric = 'accuracy' if self.problem_type == 'classification' else 'mae'
            
        # Get parameters (child classes will override this)
        params = {}
        
        # Get appropriate model class (child classes will override this)
        model_class = self._get_model_class()
        
        # Use the centralized cross-validation module
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
            # This is normal part of optimization, so just pass it along
            raise
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            # Return a poor score to tell Optuna this trial failed
            if self.study.direction == 'maximize':
                return float('-inf')
            else:
                return float('inf')
    
    def _get_model_class(self):
        """
        Get the model class to use for training. 
        This should be overridden by child classes.
        """
        # This will be overridden by child classes
        return None

    def train_model(self, random_state=42, metric='auto', force_type=None, n_trials=10, enable_pruning=True, test_size=0.2, split_type='random',time_col=None):
        """
        Main training function that handles the common ML workflow
        
        Parameters:
        - random_state: Random seed for reproducibility
        - metric: What to optimize for ('auto', 'accuracy', 'mae', etc.)
        - force_type: Force 'classification' or 'regression'
        - n_trials: Number of optimization trials to run (default: 10)
        - enable_pruning: Whether to enable Optuna pruning (stops bad trials early)
        """
        # Get problem type
        self.problem_type = force_type if force_type else self.detect_problem_type()
        print(f"\nProblem type: {self.problem_type.upper()}")
        
        # Split data using parent method
        X_train, X_test, y_train, y_test = self.split_data(
            random_state=random_state,
            test_size=test_size,
            split_type=split_type,
            time_col=time_col
        )
        
        print(f"Split type used: {split_type}")
        print("Train rows:", len(X_train))
        print("Test rows:", len(X_test))
    
    
        # Set default metric if auto
        if metric == 'auto':
            metric = 'accuracy' if self.problem_type == 'classification' else 'mae'
        
        # Store the optimization metric
        self.optimization_metric = metric
        
        # Determine optimization direction based on metric
        minimize_metrics = {'mae', 'mse', 'rmse'}
        direction = 'minimize' if metric in minimize_metrics else 'maximize'
        
        # Optimize hyperparameters
        print("\nOptimizing hyperparameters...")
        print(f"Optimizing for metric: {metric}")
        print(f"Optimization direction: {direction}")
        print(f"Pruning enabled: {enable_pruning}")
        
        # Set up pruning
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Number of trials before pruning starts
            n_warmup_steps=2,    # Wait for 2 CV folds before pruning
            interval_steps=1     # Check pruning after each fold
        ) if enable_pruning else optuna.pruners.NopPruner()
        
        # Create sampler with pruning
        sampler = optuna.samplers.TPESampler(seed=random_state)
        
        # Suppress Optuna's internal logging of warnings
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.study = optuna.create_study(
            direction=direction, 
            sampler=sampler,
            pruner=pruner
        )
        
        # Run optimization with suppressed warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            self.study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, random_state, metric),
                n_trials=n_trials
            )
        
        # Get best parameters and create model
        self.best_params = self.study.best_params
        print("\nBest parameters found:", self.best_params)
        
        # Create and train final model with suppressed warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            self.model = self._create_model(self.best_params, random_state)
            self.model.fit(X_train, y_train)
            
        # -------------------------------------------------
        # Persist train/test split (CRITICAL)
        # -------------------------------------------------
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        # -------------------------------------------------
        # Persist predictions
        # -------------------------------------------------
        self.y_train_pred = self.model.predict(X_train)
        self.y_test_pred  = self.model.predict(X_test)

        # -------------------------------------------------
        # Persist probabilities (classification only)
        # -------------------------------------------------
        if self.problem_type == "classification":
            self.y_train_proba = self.model.predict_proba(X_train)[:, 1]
            self.y_test_proba  = self.model.predict_proba(X_test)[:, 1]

        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics using parent method
        self.train_metrics = self.calculate_metrics(y_train, train_pred)
        self.test_metrics = self.calculate_metrics(y_test, test_pred)
        
        # Calculate feature importance using centralized module
        self.feature_importance, self.explanation = calculate_feature_importance(
            model=self.model,
            X_train=X_train,
            problem_type=self.problem_type,
            model_type=self.model_type
        )
            
        return self
    
    def predict(self, X_new: pd.DataFrame, return_proba: bool = True) -> pd.DataFrame:
        """
        Predict on new/unseen data using a trained model.
        """

        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        X = X_new.copy()

        # Apply preprocessing (NO FIT)
        if self.preprocessor:
            X = self.preprocessor.transform_data(X)

        # Classification
        if self.problem_type == "classification":
            preds = pd.DataFrame(index=X.index)

            if return_proba:
                preds["predProb"] = self.model.predict_proba(X)[:, 1]

            preds["predClass"] = self.model.predict(X)
            return preds

        # Regression
        preds = pd.DataFrame(index=X.index)
        preds["prediction"] = self.model.predict(X)
        return preds


    def _get_error_message(self, error):
        """Make error messages easy to understand"""
        common_errors = {
            'ValueError': "Error: Something's wrong with the data values",
            'TypeError': "Error: The data types don't match what we need",
            'AttributeError': "Error: We're missing something important",
            'KeyError': "Error: We can't find a column we need"
        }
        error_type = type(error).__name__
        return common_errors.get(error_type, f"An error occurred: {str(error)}")
        
    def print_metrics(self, show_plots=False):
        """Show how well our model is performing"""
        print("\n=== Model Performance ===")
        
        if self.problem_type == 'classification':
            print("\nClass Distribution:")
            unique, counts = np.unique(self.df[self.target_variable], return_counts=True)
            for class_label, count in zip(unique, counts):
                print(f"Class {class_label}: {count} samples ({count/len(self.df)*100:.1f}%)")
        
        print("\nTraining Metrics:")
        for metric, value in self.train_metrics.items():
            print(f"{metric.title()}: {value:.4f}")
        
        print("\nTest Metrics:")
        for metric, value in self.test_metrics.items():
            print(f"{metric.title()}: {value:.4f}")
        
        if self.feature_importance is not None:
            print("\nTop 5 Most Important Features:")
            print(self.feature_importance.head())
            
            if self.explanation:
                print(self.explanation)
        
        if show_plots:
            print("\nCreating visualizations...")
        
        return self
        
    def create_visualizations(self, show_plots=False):
        """Create optimization visualizations using Optuna"""
        print("- Optimization plots show scaled values (usually between 0-1)")
        print("- The metrics below show actual unscaled values")
        print("- This difference is due to data standardization during preprocessing")
        import optuna.visualization as viz
        
        if not show_plots:
            return self
        
        pio.renderers.default = "browser"
        
        try:
            # Optimization history
            history_plot = viz.plot_optimization_history(self.study)
            history_plot.show()
            
            # Parameter importance
            param_importance = viz.plot_param_importances(self.study)
            param_importance.show()
            
            # Parallel coordinate
            parallel_plot = viz.plot_parallel_coordinate(self.study)
            parallel_plot.show()
            
            # Plot pruning statistics if pruning was used
            if not isinstance(self.study.pruner, optuna.pruners.NopPruner):
                pruning_plot = viz.plot_intermediate_values(self.study)
                pruning_plot.show()
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            print("Try increasing the number of trials or checking if optimization was successful")
        
        return self