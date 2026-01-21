# Keep your imports at the top
from .DataPreprocessor import DataPreprocessor
from .RandomForestModel import RandomForestModel
from .UnifiedNeuralNetwork import UnifiedNeuralNetwork
from .LRegressionModel import LRegressionModel
from .XGBoostModel import XGBoostModel
import pandas as pd

def run_ml_pipeline(
        df,  # This will now be the *already cleaned* data
        target_column, 
        preprocessor, # Add this to accept the preprocessor object
        model_type='xgb', 
        random_seed=42, 
        metric='auto',
        n_trials=10,
        test_size=0.2,
        show_plots=False,        
        force_type=None,
        split_type="random",
        time_col=None
        ):
    """
    Trains a single ML model on PRE-PROCESSED data.
    This function no longer handles the main preprocessing pipeline itself.
    It now expects clean data and a preprocessor object to handle scaling.
    """
    # Check if our data is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame!")
    
    if split_type == "time" and time_col is None:
        raise ValueError("time_col must be provided when split_type='time'")
    
    # The 'run_multiple_models' function now handles this before calling us.
    # The 'df' we receive is already the 'clean_data'.
    
    # Map of model types to their classes
    model_classes = {
        'xgb': (XGBoostModel, "XGBoost"),
        'xgboost': (XGBoostModel, "XGBoost"),
        'rf': (RandomForestModel, "Random Forest"),
        'randomforest': (RandomForestModel, "Random Forest"),
        'nn': (UnifiedNeuralNetwork, "Neural Network"),
        'neuralnet': (UnifiedNeuralNetwork, "Neural Network"),
        'neuralnetwork': (UnifiedNeuralNetwork, "Neural Network"),
        'lr': (LRegressionModel, "Linear/Logistic Regression"),
        'linear': (LRegressionModel, "Linear/Logistic Regression"),
        'logistic': (LRegressionModel, "Linear/Logistic Regression")
    }
    
    model_type = str(model_type).lower().strip()
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type '{model_type}'. Valid types are: {list(model_classes.keys())}")
    
    ModelClass, model_name = model_classes[model_type]
    print(f"\n=== Training {model_name} Model ===")
    
    try:
        # --- SIMPLIFY MODEL INITIALIZATION ---
        # We now pass the preprocessor object directly to the model's constructor.
        # The model class itself will be responsible for using it to scale the data
        # after the train-test split.
        model = ModelClass(
            df=df,  # Use the clean data directly
            target_column=target_column, 
            preprocessor=preprocessor
        )
        
        problem_type = force_type if force_type else model.detect_problem_type()
        
        # Train and show results
        model.train_model(
            random_state=random_seed,
            metric=metric,
            force_type=problem_type,
            n_trials=n_trials,
            test_size=test_size,
            split_type=split_type,
            time_col=time_col
        )
        model.print_metrics()
        
        if show_plots:
            model.create_visualizations(show_plots=show_plots)
        
        return model
        
    except Exception as e:
        print(f"\nError training model: {str(e)}")
        raise