# EnsembleModel

from pipeline_code.BaseModel import BaseModel
from sklearn.ensemble import (
    VotingRegressor, 
    VotingClassifier, 
    StackingRegressor, 
    StackingClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np

class EnsembleModel(BaseModel):
    """
    Combines multiple models together to make better predictions.
    Supports four ensemble types:
    - voting: Simple averaging/voting of model predictions
    - weighted: Weighted averaging based on model performance
    - stacking: Trains a meta-model to combine predictions
    - boosting: Uses GradientBoosting for better performance
    """
    
    def __init__(self, df, target_variable, ensemble_type='voting', models=None, weights=None, preprocessor=None):
        """Set up the ensemble model"""
        super().__init__(df, target_variable, preprocessor)
        self.ensemble_type = ensemble_type.lower()
        self.input_models = models if models is not None else {}
        self.weights = weights
        
    def train_model(self, random_state=42, test_size=0.2, force_type=None, **kwargs):
        """Create and train the ensemble model"""
        # Make sure we have models to ensemble
        if not self.input_models:
            raise ValueError("No models provided for ensemble")
            
        # Get the problem type from the first model
        first_model = next(iter(self.input_models.values()))
        self.problem_type = force_type if force_type else first_model.problem_type
        
        # Create list of (name, model) tuples for the ensemble
        self.estimators = []
        model_names = []
        for name, model in self.input_models.items():
            if model.problem_type == self.problem_type:
                self.estimators.append((name, model.model))
                model_names.append(name)
        
        # Print what models we're using
        print(f"\n🔄 Creating {self.ensemble_type} ensemble with {len(self.estimators)} models:")
        for name in model_names:
            print(f"  - {name}")
        
        # Create the right type of ensemble model
        self._create_ensemble(random_state)
        
        # Split the data
        X_train, X_test, y_train, y_test = self.split_data(random_state, test_size)
        
        # Train the model
        print(f"\n🏋️ Training {self.ensemble_type} ensemble...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.train_metrics = self.calculate_metrics(y_train, train_pred)
        self.test_metrics = self.calculate_metrics(y_test, test_pred)
        
        # Set feature importance
        self._set_ensemble_importance()
        
        print(f"✅ {self.ensemble_type.title()} ensemble training complete!")
        return self
    
    def _create_ensemble(self, random_state=42):
        """Create the right type of ensemble model"""
        if self.problem_type == 'classification':
            if self.ensemble_type == 'voting':
                # For classification, use soft voting (probability-based)
                self.model = VotingClassifier(
                    estimators=self.estimators, # List of (name, model) tuples for voting
                    voting='soft'
                )
                
            elif self.ensemble_type == 'weighted':
                # Use provided weights or default to equal weights
                if not self.weights or len(self.weights) != len(self.estimators):
                    self.weights = [1] * len(self.estimators)
                    
                self.model = VotingClassifier(
                    estimators=self.estimators, # List of (name, model) tuples for voting
                    voting='soft',
                    weights=self.weights # How much influence each model gets in voting (higher = more influence)
                )
                
            elif self.ensemble_type == 'stacking':
                # Use logistic regression as the meta-model
                self.model = StackingClassifier(
                    estimators=self.estimators, # Base models to be combined
                    final_estimator=LogisticRegression(max_iter=1000), # Meta-model that learns how to combine base predictions
                    cv=5 # Cross-validation folds for training the meta-model
                )
                
            elif self.ensemble_type == 'boosting':
                # Use GradientBoosting for better performance
                self.model = GradientBoostingClassifier(
                    n_estimators=100,       # Number of boosting stages/trees to build
                    learning_rate=0.1,      # Shrinks contribution of each tree (smaller = more robust)
                    max_depth=4,            # Maximum depth of each tree (prevents overfitting)
                    min_samples_split=5,    # Minimum samples required to split a node
                    min_samples_leaf=3,     # Minimum samples required in a leaf node
                    subsample=0.8,          # Fraction of samples used for fitting each tree (< 1.0 = stochastic)
                    random_state=random_state  # Ensures reproducible results
                )
            
            else:
                raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
                
        else:  # regression
            if self.ensemble_type == 'voting':
                # Simple averaging of predictions
                self.model = VotingRegressor(
                    estimators=self.estimators  # List of (name, model) tuples for averaging predictions
                )
                
            elif self.ensemble_type == 'weighted':
                # Use provided weights or default to equal weights
                if not self.weights or len(self.weights) != len(self.estimators):
                    self.weights = [1] * len(self.estimators)  # Equal weights if none provided
                    
                self.model = VotingRegressor(
                    estimators=self.estimators,  # List of (name, model) tuples for averaging
                    weights=self.weights         # How much influence each model gets (higher = more influence)
                )
                
            elif self.ensemble_type == 'stacking':
                # Use linear regression as the meta-model
                self.model = StackingRegressor(
                    estimators=self.estimators,            # Base models to be combined
                    final_estimator=LinearRegression(),    # Meta-model that learns optimal combination weights
                    cv=5                                   # Cross-validation folds for training the meta-model
                )
                
            elif self.ensemble_type == 'boosting':
                # Use GradientBoosting for regression too
                self.model = GradientBoostingRegressor(
                    n_estimators=100,        # Number of boosting stages/trees to be built
                    learning_rate=0.1,       # Shrinks the contribution of each tree (smaller = more robust)
                    max_depth=4,             # Maximum depth of each decision tree (controls complexity)
                    min_samples_split=5,     # Minimum samples required to split an internal node
                    min_samples_leaf=3,      # Minimum samples required to be at a leaf node
                    subsample=0.8,           # Fraction of samples used for fitting (< 1.0 = stochastic gradient boosting)
                    random_state=random_state  # Controls randomness for reproducible results
                )
                
            else:
                raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def _set_ensemble_importance(self):
        """Create a simple feature importance for ensemble methods"""
        # Try to extract feature importance from the ensemble model if possible
        feature_importance = None
        explanation = None
        
        # For stacking, try to get importance from the final estimator
        if self.ensemble_type == 'stacking' and hasattr(self.model.final_estimator_, 'coef_'):
            try:
                # Get the feature names from the first model's X data
                X_train, _, _, _ = self.split_data()
                
                # Get coefficients from the final estimator (meta-model)
                if self.problem_type == 'classification':
                    coefficients = self.model.final_estimator_.coef_[0]
                else:
                    coefficients = self.model.final_estimator_.coef_
                
                # Create DataFrame with model names and their meta-model coefficients
                model_names = [name for name, _ in self.estimators]
                feature_importance = pd.DataFrame({
                    'feature': model_names,
                    'importance': np.abs(coefficients),
                    'coefficient': coefficients
                }).sort_values('importance', ascending=False)
                
                explanation = "Stacking importance shows how much weight the meta-model gives to each base model"
            except Exception as e:
                print(f"Warning: Could not extract stacking importance: {e}")
        
        # For boosting, get feature importance from GradientBoosting
        elif self.ensemble_type == 'boosting':
            try:
                X_train, _, _, _ = self.split_data()
                
                # GradientBoosting always has feature_importances_
                if hasattr(self.model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': self.model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    explanation = "Boosting importance shows which features contribute most to the model's decisions"
                else:
                    # Fallback (shouldn't happen with GradientBoosting)
                    feature_importance = pd.DataFrame({
                        'feature': ['Boosting_Method'],
                        'importance': [1.0]
                    })
                    explanation = "Boosting ensemble - feature importance not available"
                    
            except Exception as e:
                print(f"Warning: Could not extract boosting importance: {e}")
                # Fallback importance
                feature_importance = pd.DataFrame({
                    'feature': ['Boosting_Method'],
                    'importance': [1.0]
                })
                explanation = "Boosting ensemble - feature importance calculation failed"
        
        # Use model weights for weighted voting if available
        elif self.ensemble_type == 'weighted' and hasattr(self.model, 'weights'):
            try:
                model_names = [name for name, _ in self.estimators]
                feature_importance = pd.DataFrame({
                    'feature': model_names,
                    'importance': self.model.weights
                }).sort_values('importance', ascending=False)
                
                explanation = "Weighted voting importance shows the weight assigned to each model"
            except Exception as e:
                print(f"Warning: Could not extract weighted voting importance: {e}")
        
        # If we couldn't get meaningful feature importance, create a placeholder
        if feature_importance is None:
            # Create simple placeholder with ensemble type
            feature_importance = pd.DataFrame({
                'feature': ['Ensemble_Method'],
                'importance': [1.0]
            })
            
            # Add explanatory note based on ensemble type
            if self.ensemble_type == 'voting':
                explanation = "Voting ensemble combines predictions from all models equally"
            elif self.ensemble_type == 'weighted':
                explanation = "Weighted ensemble gives more importance to better-performing models"
            elif self.ensemble_type == 'stacking':
                explanation = "Stacking ensemble uses a meta-model to combine predictions from base models"
            elif self.ensemble_type == 'boosting':
                explanation = "Boosting ensemble builds models sequentially to correct errors"
        
        self.feature_importance = feature_importance
        self.explanation = explanation
        return self.feature_importance