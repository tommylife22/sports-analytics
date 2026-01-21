# PickleModels
import pickle
import os
import datetime
import sys
from pathlib import Path

# Add MLPipelineCode to path so pickle can find pipeline_code modules
_betting_models_root = Path(__file__).resolve().parent.parent
if str(_betting_models_root) not in sys.path:
    sys.path.insert(0, str(_betting_models_root))

def save_model(model, model_name=None, directory="saved_models", overwrite=False):
    """
    Save a trained machine learning model using pickle
    
    Parameters:
    model: The trained model to save
    model_name: Optional name for the saved model file
    directory: Folder to save the model in (will be created if it doesn't exist)
    overwrite: Whether to overwrite if file already exists (default: False)
    
    Returns:
    filename: Path to the saved model file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a filename if not provided
    if model_name is None:
        # Use model type and current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = type(model).__name__
        model_name = f"{model_type}_{timestamp}"
    
    # Make sure filename has .pkl extension
    if not model_name.endswith('.pkl'):
        model_name += '.pkl'
    
    # Full path to save the model
    filepath = os.path.join(directory, model_name)
    
    # Check if file already exists
    if os.path.exists(filepath) and not overwrite:
        print(f"File {filepath} already exists!")
        print(f"To overwrite, set overwrite=True or choose a different name.")
        
        # Automatically create a new filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{os.path.splitext(model_name)[0]}_{timestamp}.pkl"
        filepath = os.path.join(directory, new_name)
        print(f"Saving to {filepath} instead")
    
    # Save the model
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved successfully to {filepath}")
    return filepath


def load_model(filepath):
    """
    Load a saved model from a pickle file

    Parameters:
    filepath: Path to the pickle file

    Returns:
    The loaded model
    """
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        return None

    # Load the model with compatibility for older Optuna versions
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {filepath}")
        return model
    except TypeError as e:
        if "_ParzenEstimatorParameters" in str(e) or "categorical_distance_func" in str(e):
            print(f"⚠️  Warning: Model was saved with an incompatible Optuna version")
            print(f"⚠️  The hyperparameter optimization history cannot be loaded")
            print(f"⚠️  Attempting to load just the trained model...")

            # Try loading with custom unpickler that skips Optuna objects
            class OptunaCompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Skip problematic Optuna internal classes
                    if "optuna" in module and "Parzen" in name:
                        return object  # Return dummy object
                    return super().find_class(module, name)

            try:
                with open(filepath, 'rb') as file:
                    model = OptunaCompatUnpickler(file).load()
                print(f"✓ Model loaded (without optimization history) from {filepath}")
                print(f"ℹ️  Recommendation: Re-train and save the model with current environment")
                return model
            except Exception as e2:
                print(f"❌ Could not load model: {e2}")
                print(f"❌ Please re-train the model with your current environment")
                return None
        else:
            # Re-raise if it's a different error
            raise


def save_multiple_models(models_dict, directory="saved_models", overwrite=False):
    """
    Save multiple models at once
    
    Parameters:
    models_dict: Dictionary of {model_name: model_object}
    directory: Folder to save models in
    overwrite: Whether to overwrite if files already exist (default: False)
    
    Returns:
    saved_paths: Dictionary of {model_name: saved_filepath}
    """
    saved_paths = {}
    
    print(f"Saving {len(models_dict)} models...")
    
    for name, model in models_dict.items():
        filepath = save_model(model, name, directory, overwrite)
        saved_paths[name] = filepath
    
    print(f"All models saved to {directory}/")
    return saved_paths