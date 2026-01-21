# ==============================================================================
# Correctly import the high-level function from your DataPreprocessor.py module
from pipeline_code.DataPreprocessor import preprocess_data 
from pipeline_code.mainMLM import run_ml_pipeline
from pipeline_code.ModelAveraging import run_model_multiple_times

# ==============================================================================
# THE MAIN FUNCTION
# ==============================================================================

def run_multiple_models(
        df=None,
        file_path=None, 
        target_column=None,
        model_types=['xgb', 'rf', 'nn', 'lr'],
        random_seed=42,
        metric='auto',
        n_trials=100,
        test_size=0.2,
        show_plots=False,
        force_type=None,
        n_runs=1,
        skip_preprocessing=False
        ):
    """
    Runs multiple ML models, orchestrating preprocessing and training.
    """
    
    # Nice display names for our models 
    model_names = {
        'nn': 'NeuralNet',
        'rf': 'RandomForest',
        'xgb': 'XGBoost',
        'lr': 'LRegression'
    }

    # --- Step 1: Prepare the data using the dedicated function ---
    # This single, clean call handles all the complex logic.
    clean_data, preprocessor = preprocess_data(
        df=df,
        file_path=file_path,
        target_column=target_column,
        skip_preprocessing=skip_preprocessing
    )
    
    # --- Step 2: Train each model using the prepared data ---
    models = {}
    
    for model_type in model_types:
        display_name = model_names.get(model_type, model_type)
        print(f"\n--- Training {display_name} ---")
        
        try:
            # Train the model
            model = run_ml_pipeline(
                df=clean_data,
                target_column=target_column,
                preprocessor=preprocessor,  # Pass the preprocessor object for correct scaling
                model_type=model_type,
                random_seed=random_seed,
                metric=metric,
                n_trials=n_trials,
                test_size=test_size,
                show_plots=show_plots,
                force_type=force_type
            )
            
            if n_runs > 1:
                try:
                    run_model_multiple_times(model, n_runs=n_runs, random_seed=random_seed)
                except Exception as e:
                    print(f"Warning: Error in model averaging: {e}. Using single model results.")
                    
            models[display_name] = model
            print(f"✅ {display_name} training completed!")
            
        except Exception as e:
            print(f"❌ Error training {display_name}: {str(e)}")
    
    return models