# ModelAveraging

def run_model_multiple_times(model_object, n_runs=5, random_seed=42):
    """
    Run a model multiple times and average the results for more stable performance.
    
    Parameters:
    -----------
    model_object : BaseModel instance
        The model object with hyperparameters already tuned
    n_runs : int
        Number of times to run the model with different random seeds
    random_seed : int
        Base random seed (will be incremented for each run)
    
    Returns:
    --------
    dict
        Dictionary containing averaged metrics and aggregated feature importance
    """
    import pandas as pd
    import numpy as np
    
    print(f"\n🔄 Running model {n_runs} times for stable metrics...")
    print(f"   Model type: {model_object.__class__.__name__}")
    print(f"   Problem type: {model_object.problem_type}")
    
    # Get the optimization metric directly from the model
    if hasattr(model_object, 'optimization_metric'):
        optimization_metric = model_object.optimization_metric
    else:
        # Fallback to default metric based on problem type
        optimization_metric = 'accuracy' if model_object.problem_type == 'classification' else 'mae'
    
    print(f"   Optimization metric: {optimization_metric}")
    
    # Check if best_params exists
    if not hasattr(model_object, 'best_params') or model_object.best_params is None:
        print("   Warning: No best_params found. Using default parameters.")
        best_params = {}
    else:
        best_params = model_object.best_params
        print(f"   Using best parameters: {best_params}")
    
    # Store results from each run
    train_metrics_list = []
    test_metrics_list = []
    feature_importance_list = []
    
    # Keep track of the original model
    original_model = model_object.model
    original_train_metrics = model_object.train_metrics
    original_test_metrics = model_object.test_metrics
    original_feature_importance = model_object.feature_importance
    
    # Get the best hyperparameters found during optimization
    best_params = model_object.best_params
    problem_type = model_object.problem_type
    
    # Use existing results from the original model
    if hasattr(model_object, 'train_metrics') and hasattr(model_object, 'test_metrics'):
        # Convert metrics to the expected format
        original_train_metrics_formatted = {}
        for k, v in model_object.train_metrics.items():
            original_train_metrics_formatted[f"train_{k}"] = v
            
        original_test_metrics_formatted = {}
        for k, v in model_object.test_metrics.items():
            original_test_metrics_formatted[f"test_{k}"] = v
        
        # Add original results as the first run
        train_metrics_list.append(original_train_metrics_formatted)
        test_metrics_list.append(original_test_metrics_formatted)
        
        if hasattr(model_object, 'feature_importance') and model_object.feature_importance is not None:
            feature_importance_list.append(model_object.feature_importance)
        
        # Start from run 2 (index 1) since we already have run 1
        start_index = 1
        print(f"   ✅ Using existing results from original training (seed: {random_seed})")
        
        # Display original metrics using the optimization metric
        if optimization_metric in model_object.test_metrics:
            print(f"      Original {optimization_metric}: {model_object.test_metrics[optimization_metric]:.4f}")
        else:
            # Fallback to default metrics
            if problem_type == 'classification':
                print(f"      Original accuracy: {model_object.test_metrics['accuracy']:.4f}")
            else:
                print(f"      Original RMSE: {model_object.test_metrics['rmse']:.4f}")
    else:
        # Start from run 1 if no existing results
        start_index = 0
        print("   ⚠️ No existing results found. Running all iterations.")
    
    # Run the model additional times with different random seeds
    for i in range(start_index, n_runs):
        current_seed = random_seed + i
        print(f"   Run {i+1}/{n_runs} (seed: {current_seed})...", end="")
        
        try:
            # Split data with new random seed
            X_train, X_test, y_train, y_test = model_object.split_data(random_state=current_seed)
            
            # Create and train model with best params but new random seed
            try:
                # Make sure best_params exists
                if best_params is None:
                    print(" Error: No best parameters found. Skipping this run.")
                    continue
                    
                new_model = model_object._create_model(best_params, current_seed)
                new_model.fit(X_train, y_train)
            except Exception as e:
                print(f" Error creating/training model: {e}")
                continue
            
            # Make predictions
            train_pred = new_model.predict(X_train)
            test_pred = new_model.predict(X_test)
            
            # Calculate metrics - convert to the format expected by later code
            train_metrics = {}
            for k, v in model_object.calculate_metrics(y_train, train_pred).items():
                train_metrics[f"train_{k}"] = v
                
            test_metrics = {}
            for k, v in model_object.calculate_metrics(y_test, test_pred).items():
                test_metrics[f"test_{k}"] = v
            
            # Get feature importance if available
            try:
                # First try using the model's own method if it exists
                if hasattr(model_object, '_get_feature_importance'):
                    feature_importance, _ = model_object._get_feature_importance(new_model, X_train)
                    feature_importance_list.append(feature_importance)
                # Otherwise try direct import
                else:
                    try:
                        from pipeline_code.FeatureImportance import calculate_feature_importance
                        feature_importance, _ = calculate_feature_importance(
                            model=new_model,
                            X_train=X_train,
                            problem_type=model_object.problem_type,
                            model_type=model_object.model_type
                        )
                        feature_importance_list.append(feature_importance)
                    except ImportError:
                        print(f" Warning: FeatureImportance module not found")
            except Exception as e:
                print(f" Warning: Could not get feature importance: {e}")
            
            # Store metrics
            train_metrics_list.append(train_metrics)
            test_metrics_list.append(test_metrics)
            
            # Display progress using the optimization metric
            test_metric_key = f"test_{optimization_metric}"
            if test_metric_key in test_metrics:
                print(f" done! ({optimization_metric}: {test_metrics[test_metric_key]:.4f})")
            else:
                # Fallback to default display
                if problem_type == 'classification':
                    print(f" done! (accuracy: {test_metrics['test_accuracy']:.4f})")
                else:
                    print(f" done! (RMSE: {test_metrics['test_rmse']:.4f})")
                
        except Exception as e:
            print(f" Error in run {i+1}: {e}")
    
    # Aggregate the metrics across all runs
    avg_train_metrics = {}
    avg_test_metrics = {}
    
    # Process train metrics
    if train_metrics_list:
        for key in train_metrics_list[0].keys():
            # Make sure all metrics have this key
            values = [m.get(key) for m in train_metrics_list if key in m]
            if values:
                avg_train_metrics[key] = sum(values) / len(values)
                avg_train_metrics[f"{key}_std"] = np.std(values)
    
    # Process test metrics  
    if test_metrics_list:
        for key in test_metrics_list[0].keys():
            # Make sure all metrics have this key
            values = [m.get(key) for m in test_metrics_list if key in m]
            if values:
                avg_test_metrics[key] = sum(values) / len(values)
                avg_test_metrics[f"{key}_std"] = np.std(values)
    
    # Aggregate feature importance
    avg_feature_importance = None
    if feature_importance_list:
        # Get all unique feature names across all runs
        all_features = set()
        for df in feature_importance_list:
            all_features.update(df['feature'])
        
        # Average the importance values for each feature
        agg_data = []
        for feature in all_features:
            importance_values = []
            for df in feature_importance_list:
                feature_rows = df[df['feature'] == feature]
                if not feature_rows.empty and 'importance' in df.columns:
                    importance_values.append(feature_rows['importance'].values[0])
            
            if importance_values:
                avg_importance = sum(importance_values) / len(importance_values)
                std_importance = np.std(importance_values)
                agg_data.append({
                    'feature': feature, 
                    'importance': avg_importance,
                    'importance_std': std_importance
                })
        
        if agg_data:
            avg_feature_importance = pd.DataFrame(agg_data)
            avg_feature_importance = avg_feature_importance.sort_values(
                'importance', ascending=False
            ).reset_index(drop=True)
    
    # Restore original model and metrics
    model_object.model = original_model
    model_object.train_metrics = original_train_metrics
    model_object.test_metrics = original_test_metrics
    model_object.feature_importance = original_feature_importance
    
    # Attach averaged metrics to the model object
    model_object.avg_train_metrics = avg_train_metrics
    model_object.avg_test_metrics = avg_test_metrics
    model_object.avg_feature_importance = avg_feature_importance
    model_object.n_runs = n_runs
    
    # Store the optimization metric for reference
    model_object.optimization_metric = optimization_metric
    
    # Return result dictionary
    result = {
        'avg_train_metrics': avg_train_metrics,
        'avg_test_metrics': avg_test_metrics,
        'avg_feature_importance': avg_feature_importance,
        'n_runs': n_runs,
        'optimization_metric': optimization_metric
    }
    
    print(f"\n✅ Model averaging completed with {n_runs} runs")
    print(f"   Optimized for: {optimization_metric}")
    
    return result