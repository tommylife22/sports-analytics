# FeatureImportance

import pandas as pd
import numpy as np

def calculate_feature_importance(model, X_train, problem_type, model_type):
    """
    Calculate feature importance for any model type
    
    Parameters:
    -----------
    model : model object
        The trained model to calculate feature importance for
    X_train : DataFrame
        Training features used to train the model
    problem_type : str
        Either 'classification' or 'regression'
    model_type : str
        The type of model (e.g., 'xgboost', 'randomforest', 'linear', 'neuralnetwork')
        
    Returns:
    --------
    DataFrame
        DataFrame containing feature importance information
    str
        Explanatory note about the feature importance
    """
    # Select the appropriate calculation method based on model type
    if model_type.lower() in ['xgboost', 'xgb']:
        return _calculate_xgboost_importance(model, X_train)
    elif model_type.lower() in ['randomforest', 'rf']:
        return _calculate_randomforest_importance(model, X_train)
    elif model_type.lower() in ['linear', 'logistic', 'lregression', 'lr']:
        return _calculate_linear_importance(model, X_train, problem_type)
    elif model_type.lower() in ['neuralnetwork', 'nn', 'unifiedneuralnetwork']:
        return _calculate_nn_importance(model, X_train)
    else:
        return _calculate_generic_importance(model, X_train)

def _calculate_xgboost_importance(model, X_train):
    """Calculate feature importance for XGBoost models"""
    # Get standard feature importance (weight)
    weight_importance = model.feature_importances_
    
    # Get gain-based feature importance
    booster = model.get_booster()
    gain_importance = booster.get_score(importance_type='gain')
    
    # Store raw gain values for sorting
    raw_gain = [gain_importance.get(f, 0) for f in X_train.columns]
    
    # Calculate percentage of total gain (handle zero case)
    total_gain = sum(raw_gain)
    if total_gain > 0:
        percentages = [(g / total_gain * 100) for g in raw_gain]
    else:
        percentages = [0.0] * len(raw_gain)
    
    # Format gain values for readability
    formatted_gain = [f"{g:,.2f}" if g < 1000 else f"{int(g):,}" for g in raw_gain]
    
    # Create DataFrame with all metrics
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'weight': weight_importance.round(4),
        'gain': formatted_gain,
        'percent': [f"{p:.2f}%" for p in percentages]
    })
    
    # Sort by raw gain values (not the formatted strings)
    gain_order = sorted(range(len(raw_gain)), key=lambda i: raw_gain[i], reverse=True)
    importance_df = importance_df.iloc[gain_order].reset_index(drop=True)
    
    # Add explanation note
    explanation = """
    XGBoost Importance Explanation:
    - weight: How frequently the feature is used in the model's trees
    - gain: How much prediction improvement the feature provides
    - percent: The feature's contribution relative to all features
    """
    
    return importance_df, explanation

def _calculate_randomforest_importance(model, X_train):
    """Calculate feature importance for Random Forest models"""
    # Get feature importance values
    importance_values = model.feature_importances_
    
    # Calculate percentage of total importance (handle zero case)
    total_importance = importance_values.sum()
    if total_importance > 0:
        percentages = [(imp / total_importance * 100) for imp in importance_values]
    else:
        percentages = [0.0] * len(importance_values)
    
    # Create DataFrame with both raw importance and percentage
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance_values.round(6),
        'percent': [f"{p:.2f}%" for p in percentages]
    }).sort_values('importance', ascending=False)
    
    # Add explanatory note
    explanation = """
    Random Forest Importance Explanation:
    - importance: Measures how much each feature reduces impurity across all trees
    - Finding the best split for a node that results in the most homogeneous (or pure) subsets
    """
    
    return importance_df, explanation

def _calculate_linear_importance(model, X_train, problem_type):
    """Calculate feature importance for Linear/Logistic Regression models"""
    if problem_type == 'classification':
        # For logistic regression, keep the sign of coefficients
        importance = model.coef_[0]
    else:
        # For linear regression, keep the sign of coefficients
        importance = model.coef_
    
    # Calculate percentage based on absolute values (handle zero case)
    abs_importance = np.abs(importance)
    total_abs_importance = abs_importance.sum()
    if total_abs_importance > 0:
        percentages = [(abs_imp / total_abs_importance * 100) for abs_imp in abs_importance]
    else:
        percentages = [0.0] * len(abs_importance)
        
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': importance.round(6),
        'percent': [f"{p:.2f}%" for p in percentages]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    # Add explanatory note
    model_type = "Logistic Regression" if problem_type == 'classification' else "Linear Regression"
    explanation = f"""
    {model_type} Importance Explanation:
    - coefficient: Actual model coefficients (positive/negative shows direction)
    - percent: Relative contribution based on coefficient magnitude
    - Features are sorted by absolute magnitude
    """
    
    return importance_df, explanation

def _calculate_nn_importance(model, X_train):
    import shap
    """
    Calculates SHAP values for Scikit-learn MLPs using a faster configuration 
    for KernelExplainer. This function automatically detects if the model is a 
    classifier or a regressor and will not crash.
    """
    try:
        # --- Configuration for Speed ---
        background_size = 25
        explain_size = 50
        background_data = shap.sample(X_train, min(background_size, len(X_train)))
        data_to_explain = shap.sample(X_train, min(explain_size, len(X_train)))

        # --- AUTOMATIC MODEL TYPE DETECTION ---
        # This is the key change. We check if the model has the '.predict_proba'
        # method. If it does, it's a classifier. If not, it's a regressor.
        if hasattr(model, 'predict_proba'):
            print("Classifier detected. Using model.predict_proba for SHAP.")
            prediction_function = model.predict_proba
        else:
            print("Regressor detected. Using model.predict for SHAP.")
            prediction_function = model.predict
        
        # Create the explainer with the correct prediction function
        explainer = shap.KernelExplainer(prediction_function, background_data)
        
        shap_values = explainer.shap_values(data_to_explain)
        
        # --- Processing the Results ---
        # For classification, shap_values is a list. We select the positive class's values.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Calculate the mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Squeeze the result to prevent the 1-dimensional array error for Pandas
        mean_shap = mean_shap.squeeze()
        
        # Create the final DataFrame
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'shap_importance': mean_shap,
        }).sort_values('shap_importance', ascending=False)
        
        explanation = "SHAP KernelExplainer (Faster Settings): An approximation of feature importance based on a small data sample."
        return importance_df, explanation
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}. Cannot calculate feature importance with this method.")
        return pd.DataFrame(), str(e)
    
def _calculate_generic_importance(model, X_train):
    """
    Try to calculate feature importance for any model with a feature_importances_ attribute
    or a coef_ attribute
    """
    if hasattr(model, 'feature_importances_'):
        # Models like tree-based models, GradientBoosting, etc.
        importance_values = model.feature_importances_
        
        # Calculate percentage (handle zero case)
        total_importance = importance_values.sum()
        if total_importance > 0:
            percentages = [(imp / total_importance * 100) for imp in importance_values]
        else:
            percentages = [0.0] * len(importance_values)

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance_values.round(6),
            'percent': [f"{p:.2f}%" for p in percentages]
        }).sort_values('importance', ascending=False)

        explanation = """
        Generic Feature Importance:
        - importance: The relative importance of each feature
        - percent: Percentage contribution to the model
        """
        
    elif hasattr(model, 'coef_'):
        # Linear models
        if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
            # Multiclass classification case
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            # Binary classification or regression
            importance = model.coef_.flatten()
            
        # Calculate percentage (handle zero case)
        abs_importance = np.abs(importance)
        total_abs_importance = abs_importance.sum()
        if total_abs_importance > 0:
            percentages = [(abs_imp / total_abs_importance * 100) for abs_imp in abs_importance]
        else:
            percentages = [0.0] * len(abs_importance)

        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': importance.round(6),
            'percent': [f"{p:.2f}%" for p in percentages]
        }).sort_values('coefficient', key=abs, ascending=False)

        explanation = """
        Coefficient-based Feature Importance:
        - coefficient: Model coefficients (positive/negative shows direction)
        - percent: Relative contribution based on coefficient magnitude
        """
        
    else:
        # Try to use a generic permutation importance approach
        try:
            from sklearn.inspection import permutation_importance
            
            # For classification, use predict_proba if available
            if hasattr(model, 'predict_proba'):
                r = permutation_importance(model, X_train, 
                                         y_pred=model.predict_proba(X_train)[:, 1], 
                                         n_repeats=5, random_state=42)
            else:
                # For regression or models without predict_proba
                r = permutation_importance(model, X_train, 
                                         y_pred=model.predict(X_train), 
                                         n_repeats=5, random_state=42)
                
            importance_values = r.importances_mean

            # Calculate percentage (handle zero case)
            total_importance = importance_values.sum()
            if total_importance > 0:
                percentages = [(imp / total_importance * 100) for imp in importance_values]
            else:
                percentages = [0.0] * len(importance_values)
            
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importance_values.round(6),
                'percent': [f"{p:.2f}%" for p in percentages]
            }).sort_values('importance', ascending=False)
            
            explanation = """
            Permutation Feature Importance:
            - importance: How much model performance decreases when a feature is randomly shuffled
            - percent: Relative importance compared to other features
            - Higher values indicate more important features
            """
            
        except Exception as e:
            print(f"Warning: Could not calculate permutation importance: {e}")
            
            # No known feature importance method
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.ones(len(X_train.columns)) / len(X_train.columns)
            })
            explanation = "Feature importance not available for this model type."
    
    return importance_df, explanation