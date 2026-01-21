# EnsembleMethods
# A complete module for creating ensembles from pre-trained models

import pandas as pd

def create_ensemble_from_models(
        df,
        target_column,
        models_dict,
        ensemble_type='voting',
        weights=None,
        random_seed=42,
        test_size=0.2,
        save_ensemble=False,
        save_directory="saved_models",
        show_plots=False):
    """
    Create an ensemble model from pre-trained models in a dictionary
    
    Parameters:
    -----------
    df : pandas.DataFrame - DataFrame with features and target, must be the same data used to train models
    target_column : str - Name of the target column to predict
    models_dict : dict - Dictionary of {model_name: model_object} with pre-trained models
    ensemble_type : str, optional - Type of ensemble to create: 'voting', 'weighted', 'stacking', or 'boosting'. Default: 'voting'
    weights : list, optional - List of weights for weighted ensemble (must match length of models_dict)
    random_seed : int, optional - Random seed for reproducibility. Default: 42
    test_size : float, optional - Proportion of data to use for testing. Default: 0.2
    save_ensemble : bool, optional - Whether to save the ensemble model. Default: False
    save_directory : str, optional - Directory to save model in. Default: "saved_models"
    show_plots : bool, optional - Whether to display performance plots. Default: False
        
    Returns:
    --------
    EnsembleModel
        The trained ensemble model
    """
    # Import inside the function to avoid circular imports
    from pipeline_code.EnsembleModel import EnsembleModel
    from PickleModels import save_model
    
    print("🔍 Starting ensemble model creation process")
    print(f"📊 Using {len(models_dict)} pre-trained models: {', '.join(models_dict.keys())}")
    
    # Create and train the ensemble model
    print(f"🔄 Creating {ensemble_type} ensemble model")
    
    # If using weighted ensemble with no weights provided, create equal weights
    if ensemble_type == 'weighted' and weights is None:
        weights = [1] * len(models_dict)
        print(f"  Using equal weights: {weights}")
    
    # Create the ensemble model
    ensemble = EnsembleModel(
        df=df,
        target_variable=target_column,
        ensemble_type=ensemble_type,
        models=models_dict,
        weights=weights
    )
    
    # Train the ensemble model
    ensemble.train_model(random_state=random_seed, test_size=test_size)
    
    # Check if regular models use averaging
    sample_model = next(iter(models_dict.values()))
    has_avg_metrics = hasattr(sample_model, 'avg_train_metrics') and sample_model.avg_train_metrics is not None
    
    # Fix ensemble model if needed to make it compatible with averaged models
    if has_avg_metrics:
        # Add averaged metrics attributes to ensemble model
        fix_ensemble_model(sample_model, ensemble)
    
    # Print performance metrics
    ensemble.print_metrics()
    if show_plots:
        ensemble.create_visualizations(show_plots=show_plots)
    
    # Save the ensemble if requested
    if save_ensemble:
        print(f"💾 Saving ensemble model to {save_directory}")
        ensemble_path = save_model(
            ensemble, 
            f"ensemble_{ensemble_type}", 
            directory=save_directory
        )
        print(f"✅ Ensemble model saved to {ensemble_path}")
    
    # Add the ensemble to the models dictionary for comparison
    all_models = {
        **models_dict,
        f"ensemble_{ensemble_type}": ensemble
    }
    
    # Display comparison of all models
    from pipeline_code.ModelMetricDisplays import compare_model_metrics
    compare_model_metrics(all_models)
    
    print("🎉 Ensemble creation complete!")
    return ensemble

def compare_ensemble_types(
        df,
        target_column,
        models_dict,
        ensemble_types=['voting', 'weighted', 'stacking', 'boosting'],
        random_seed=42,
        test_size=0.2,
        save_ensembles=True,
        save_directory="saved_models",
        use_performance_weights=True):
    """
    Create and compare different types of ensemble models from pre-trained models
    
    Parameters:
    -----------
    df : pandas.DataFrame - DataFrame with features and target
    target_column : str - Name of the target column to predict
    models_dict : dict - Dictionary of {model_name: model_object} with pre-trained models
    ensemble_types : list, optional - List of ensemble types to create. Default: ['voting', 'weighted', 'stacking', 'boosting']
    random_seed : int, optional - Random seed for reproducibility. Default: 42
    test_size : float, optional - Proportion of data to use for testing. Default: 0.2
    save_ensembles : bool, optional - Whether to save the ensemble models. Default: True
    save_directory : str, optional - Directory to save models in. Default: "saved_models"
    use_performance_weights : bool, optional - If True, calculates weights based on model performance for weighted ensemble. If False, uses position-based weights. Default: True
        
    Returns:
    --------
    dict
        Dictionary containing all ensemble models
    """
    # Import inside the function to avoid circular imports
    from pipeline_code.EnsembleModel import EnsembleModel
    from PickleModels import save_multiple_models
    
    print("🔄 Starting ensemble comparison")
    print(f"📊 Using {len(models_dict)} pre-trained models")
    
    # Check if models use averaging
    sample_model = next(iter(models_dict.values()))
    using_averaged_metrics = hasattr(sample_model, 'avg_test_metrics') and sample_model.avg_test_metrics is not None
    
    # Create each type of ensemble
    ensembles = {}
    
    for ensemble_type in ensemble_types:
        print(f"🔄 Creating {ensemble_type} ensemble")
        
        # For weighted ensemble, handle weights
        weights = None
        if ensemble_type == 'weighted':
            if use_performance_weights:
                # Calculate weights based on model performance
                print("  Using performance-based weights")
                
                # Determine problem type from first model
                first_model = next(iter(models_dict.values()))
                problem_type = first_model.problem_type
                
                # Auto-select performance metric
                if problem_type == 'classification':
                    performance_metric = 'test_accuracy'
                else:
                    performance_metric = 'test_r2'
                
                # Calculate performance-based weights
                weights = []
                model_performance = {}
                
                # Get performance values for each model
                for name, model in models_dict.items():
                    # Get performance based on whether we're using averaged metrics
                    if using_averaged_metrics:
                        if hasattr(model, 'avg_test_metrics') and performance_metric in model.avg_test_metrics:
                            perf_value = model.avg_test_metrics[performance_metric]
                        else:
                            print(f"  Warning: Metric '{performance_metric}' not found for {name}, using default weight")
                            perf_value = 1.0
                    else:
                        # Try to get the performance metric from regular metrics
                        # Need to strip 'test_' from the metric name
                        regular_metric = performance_metric.replace('test_', '')
                        if hasattr(model, 'test_metrics') and regular_metric in model.test_metrics:
                            perf_value = model.test_metrics[regular_metric]
                        else:
                            print(f"  Warning: Metric '{regular_metric}' not found for {name}, using default weight")
                            perf_value = 1.0
                    
                    model_performance[name] = perf_value
                
                # For metrics where higher is better (accuracy, r2)
                if performance_metric in ['test_accuracy', 'test_r2', 'test_f1', 'test_precision', 'test_recall']:
                    # Handle potentially negative R2 values
                    min_val = min(model_performance.values())
                    if min_val < 0:
                        shifted_values = {name: value - min_val + 0.01 for name, value in model_performance.items()}
                        model_performance = shifted_values
                    
                    # Calculate weights (higher performance = higher weight)
                    for name in models_dict.keys():
                        weights.append(model_performance[name])
                
                # For metrics where lower is better (mae, mse, rmse)
                else:
                    # Calculate weights (lower values = higher weights, so invert)
                    for name in models_dict.keys():
                        # Avoid division by zero
                        if model_performance[name] <= 0:
                            weights.append(1.0)
                        else:
                            weights.append(1.0 / model_performance[name])
                
                # Normalize weights to sum to number of models
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w * len(weights) / total_weight for w in weights]
                else:
                    weights = [1.0] * len(models_dict)
                
                # Print out the weights
                print("  Model weights based on performance:")
                for name, weight in zip(models_dict.keys(), weights):
                    print(f"    {name}: {weight:.2f}")
                
            else:
                # Use position-based weights
                model_names = list(models_dict.keys())
                weights = [len(model_names) - i for i in range(len(model_names))]
                print(f"  Using position-based weights: {weights}")
        
        # Create and train the ensemble
        ensemble = EnsembleModel(
            df=df,
            target_variable=target_column,
            ensemble_type=ensemble_type,
            models=models_dict,
            weights=weights
        )
        
        # Train the ensemble model
        ensemble.train_model(random_state=random_seed, test_size=test_size)
        
        # Add to our collection
        ensemble_name = f"ensemble_{ensemble_type}"
        ensembles[ensemble_name] = ensemble
    
    # Fix ensemble models if needed for comparison with averaged models
    if using_averaged_metrics:
        for name, ensemble in ensembles.items():
            fix_ensemble_model(sample_model, ensemble)
    
    # Combine with original models for comparison
    all_models = {**models_dict, **ensembles}
    
    # Compare all models
    from pipeline_code.ModelMetricDisplays import compare_model_metrics
    comparison = compare_model_metrics(all_models)
    
    # Save ensemble models if requested
    if save_ensembles:
        print(f"💾 Saving ensemble models to {save_directory}")
        save_multiple_models(ensembles, directory=save_directory)
        print(f"✅ Ensemble models saved to {save_directory}")
    
    print("✅ Ensemble comparison complete!")
    return ensembles

def create_boosting_ensemble(df, target_column, base_model, n_estimators=50, learning_rate=0.1, random_seed=42, save_ensemble=True):
    """
    Create a boosting ensemble with a specific base model
    
    Parameters:
    -----------
    df : pandas.DataFrame - DataFrame with features and target
    target_column : str - Name of the target column to predict
    base_model : model object - The base model to use for boosting
    n_estimators : int, optional - Number of boosting stages. Default: 50
    learning_rate : float, optional - Learning rate shrinks the contribution of each classifier. Default: 0.1
    random_seed : int, optional - Random seed for reproducibility. Default: 42
    save_ensemble : bool, optional - Whether to save the ensemble model. Default: True
    
    Returns:
    --------
    EnsembleModel
        The trained boosting ensemble model
    """
    # Import inside the function to avoid circular imports
    from pipeline_code.EnsembleModel import EnsembleModel
    
    print("🔄 Creating boosting ensemble")
    print(f"📊 Using {base_model.__class__.__name__} as base model")
    print(f"📈 Ensemble parameters: n_estimators={n_estimators}, learning_rate={learning_rate}")
    
    # Create a dictionary with the base model
    model_name = base_model.__class__.__name__.replace('Model', '')
    models_dict = {f"{model_name}_base": base_model}
    
    # Create the ensemble model with boosting type
    ensemble = create_ensemble_from_models(
        df=df,
        target_column=target_column,
        models_dict=models_dict,
        ensemble_type='boosting',
        random_seed=random_seed,
        save_ensemble=save_ensemble
    )
    
    return ensemble

# Helper function to fix ensemble models to work with averaged models
def fix_ensemble_model(sample_model, ensemble_model):
    """
    Add averaged metrics attributes to ensemble models to make them
    compatible with averaged regular models.
    
    Parameters:
    -----------
    sample_model : model object - A sample regular model that has averaged metrics
    ensemble_model : EnsembleModel - The ensemble model to fix
    """
    # Add averaged metrics attributes
    ensemble_model.avg_train_metrics = {}
    ensemble_model.avg_test_metrics = {}
    
    # Copy train metrics with standard deviation of 0
    for key, value in ensemble_model.train_metrics.items():
        # Convert to the format used by averaged metrics
        avg_key = f"train_{key}"
        ensemble_model.avg_train_metrics[avg_key] = value
        ensemble_model.avg_train_metrics[f"{avg_key}_std"] = 0.0
    
    # Copy test metrics with standard deviation of 0
    for key, value in ensemble_model.test_metrics.items():
        # Convert to the format used by averaged metrics
        avg_key = f"test_{key}"
        ensemble_model.avg_test_metrics[avg_key] = value
        ensemble_model.avg_test_metrics[f"{avg_key}_std"] = 0.0
    
    # Add n_runs attribute for consistency
    ensemble_model.n_runs = 1
    
    # Add placeholder feature importance if needed
    if not hasattr(ensemble_model, 'avg_feature_importance'):
        ensemble_model.avg_feature_importance = ensemble_model.feature_importance