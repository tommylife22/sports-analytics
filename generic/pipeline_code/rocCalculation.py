# rocCalculation

def add_auc_to_models(models, n_runs=1, include_train=True):
    """
    Calculate and add AUC metric to classification models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model names and model objects
    n_runs : int
        Number of runs that were used for averaging
    include_train : bool
        Whether to calculate AUC for training data too (default: True)
        
    Returns:
    --------
    dict
        Same dictionary with AUC metrics added
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    
    # Check if we have classification models
    sample_model = next(iter(models.values()))
    if sample_model.problem_type != 'classification':
        print("AUC is only applicable for classification problems")
        return models
    
    # Process each model
    for name, model in models.items():
        # Check if this model has averaged metrics (was run multiple times)
        has_avg_metrics = hasattr(model, 'avg_test_metrics') and model.avg_test_metrics is not None
        
        if has_avg_metrics:
            # For models with averaging, calculate AUC across multiple runs
            test_auc_scores = []
            train_auc_scores = []
            
            # Get the number of runs that were used
            actual_n_runs = model.n_runs if hasattr(model, 'n_runs') else n_runs
            base_seed = 42
            
            # Run predictions multiple times
            for i in range(actual_n_runs):
                current_seed = base_seed + i
                
                # Use the same random state that was used for original model evaluation
                X_train, X_test, y_train, y_test = model.split_data(random_state=current_seed)
                
                # Check if binary classification
                if len(np.unique(y_test)) == 2:
                    # Get probability predictions if possible
                    if hasattr(model.model, "predict_proba"):
                        # Calculate AUC for test data
                        y_test_score = model.model.predict_proba(X_test)[:, 1]
                        test_auc_score = roc_auc_score(y_test, y_test_score)
                        test_auc_scores.append(test_auc_score)
                        
                        # Calculate AUC for training data if requested
                        if include_train:
                            y_train_score = model.model.predict_proba(X_train)[:, 1]
                            train_auc_score = roc_auc_score(y_train, y_train_score)
                            train_auc_scores.append(train_auc_score)
                    else:
                        print(f"Model {name} doesn't support predict_proba, skipping AUC")
                        break
                else:
                    print(f"AUC for multi-class requires additional handling")
                    break
            
            # Calculate average AUC and standard deviation for test data
            if test_auc_scores:
                avg_test_auc = sum(test_auc_scores) / len(test_auc_scores)
                std_test_auc = np.std(test_auc_scores) if len(test_auc_scores) > 1 else 0
                
                # Add to averaged metrics
                model.avg_test_metrics['auc'] = avg_test_auc
                model.avg_test_metrics['auc_std'] = std_test_auc
                
                print(f"{name} Average Test AUC: {avg_test_auc:.4f} (±{std_test_auc:.4f})")
            
            # Calculate average AUC and standard deviation for train data
            if train_auc_scores:
                avg_train_auc = sum(train_auc_scores) / len(train_auc_scores)
                std_train_auc = np.std(train_auc_scores) if len(train_auc_scores) > 1 else 0
                
                # Add to averaged metrics
                model.avg_train_metrics['auc'] = avg_train_auc
                model.avg_train_metrics['auc_std'] = std_train_auc
                
                print(f"{name} Average Train AUC: {avg_train_auc:.4f} (±{std_train_auc:.4f})")
        else:
            # For single-run models, calculate AUC once
            X_train, X_test, y_train, y_test = model.split_data(random_state=42)
            
            # Check if binary classification
            if len(np.unique(y_test)) == 2:
                # Get probability predictions if possible
                if hasattr(model.model, "predict_proba"):
                    # Calculate AUC for test data
                    y_test_score = model.model.predict_proba(X_test)[:, 1]
                    test_auc_score = roc_auc_score(y_test, y_test_score)
                    
                    # Add to test metrics
                    model.test_metrics['auc'] = test_auc_score
                    print(f"{name} Test AUC: {test_auc_score:.4f}")
                    
                    # Calculate AUC for training data if requested
                    if include_train:
                        y_train_score = model.model.predict_proba(X_train)[:, 1]
                        train_auc_score = roc_auc_score(y_train, y_train_score)
                        
                        # Add to train metrics
                        model.train_metrics['auc'] = train_auc_score
                        print(f"{name} Train AUC: {train_auc_score:.4f}")
                else:
                    print(f"Model {name} doesn't support predict_proba, skipping AUC")
            else:
                print(f"AUC for multi-class requires additional handling")
    
    return models

def plot_roc_curves(models, n_runs=1, save_path=None, figsize=(10, 8), include_train=False):
    """
    Plot ROC curves for classification models, supporting both single-run
    and multiple-run (averaged) models. Models are ordered from best to worst AUC.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model names and model objects
    n_runs : int
        Number of runs that were used for averaging
    save_path : str, optional
        If provided, save the plot to this path
    figsize : tuple, optional
        Figure size (width, height) in inches
    include_train : bool
        Whether to include ROC curves for training data (default: False)
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    
    # Check if we have classification models
    sample_model = next(iter(models.values()))
    if sample_model.problem_type != 'classification':
        print("ROC curves are only applicable for classification problems")
        return None
    
    print("\n=== ROC Curve Analysis ===")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # Get AUC values for all models to sort them
    model_aucs = {}
    
    for name, model in models.items():
        # Get AUC from metrics
        if hasattr(model, 'avg_test_metrics') and model.avg_test_metrics is not None and 'auc' in model.avg_test_metrics:
            model_aucs[name] = model.avg_test_metrics['auc']
        elif hasattr(model, 'test_metrics') and 'auc' in model.test_metrics:
            model_aucs[name] = model.test_metrics['auc']
        else:
            # If AUC not available, calculate it
            X_train, X_test, y_train, y_test = model.split_data(random_state=42)
            
            if hasattr(model.model, "predict_proba"):
                y_score = model.model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                model_aucs[name] = roc_auc
            else:
                # Skip models without predict_proba
                model_aucs[name] = 0
    
    # Sort models by their AUC values (descending)
    sorted_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)
    
    # For each model, compute and plot ROC curve
    for i, (name, auc_value) in enumerate(sorted_models):
        model = models[name]
        
        # Choose a color for this model
        color = colors[i % len(colors)]
        
        # Check if this model has averaged metrics
        has_avg_metrics = hasattr(model, 'avg_test_metrics') and model.avg_test_metrics is not None
        
        if has_avg_metrics:
            # For averaged models, plot multiple ROC curves with transparency
            actual_n_runs = model.n_runs if hasattr(model, 'n_runs') else n_runs
            base_seed = 42
            
            # Track TPR and FPR for averaging (test)
            test_tprs = []
            test_fprs = []
            mean_fpr = np.linspace(0, 1, 100)  # Common x-axis for averaging
            
            # Track TPR and FPR for averaging (train)
            train_tprs = []
            train_fprs = []
            
            # Run predictions multiple times
            for j in range(actual_n_runs):
                current_seed = base_seed + j
                X_train, X_test, y_train, y_test = model.split_data(random_state=current_seed)
                
                # Check if binary classification
                if len(np.unique(y_test)) == 2:
                    # Get probability predictions
                    if hasattr(model.model, "predict_proba"):
                        # Test data ROC
                        y_test_score = model.model.predict_proba(X_test)[:, 1]
                        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_score)
                        test_fprs.append(test_fpr)
                        test_tprs.append(test_tpr)
                        
                        # Plot individual test run with high transparency
                        if actual_n_runs <= 10:  # Only plot individual runs if there aren't too many
                            plt.plot(test_fpr, test_tpr, alpha=0.2, color=color, lw=1)
                        
                        # Training data ROC if requested
                        if include_train:
                            y_train_score = model.model.predict_proba(X_train)[:, 1]
                            train_fpr, train_tpr, _ = roc_curve(y_train, y_train_score)
                            train_fprs.append(train_fpr)
                            train_tprs.append(train_tpr)
                            
                            # Plot individual train run with high transparency
                            if actual_n_runs <= 10:
                                plt.plot(train_fpr, train_tpr, alpha=0.2, color=color, linestyle='--', lw=1)
                    else:
                        print(f"Model {name} doesn't support predict_proba, skipping")
                        break
                else:
                    print(f"ROC curves are designed for binary classification")
                    break
            
            # Calculate and plot the average TEST ROC curve
            if test_tprs:
                # Interpolate all TPRs to the same FPR grid
                interp_tprs = []
                for fpr, tpr in zip(test_fprs, test_tprs):
                    interp_tprs.append(np.interp(mean_fpr, fpr, tpr))
                    interp_tprs[-1][0] = 0.0  # Force starting at (0,0)
                
                # Calculate mean TPR and standard deviation
                mean_tpr = np.mean(interp_tprs, axis=0)
                mean_tpr[-1] = 1.0  # Force ending at (1,1)
                std_tpr = np.std(interp_tprs, axis=0)
                
                # Calculate AUC of the mean curve
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std([auc(mean_fpr, tpr) for tpr in interp_tprs])
                
                # Use the same AUC value that's in the model's metrics
                if 'auc' in model.avg_test_metrics:
                    mean_auc = model.avg_test_metrics['auc']
                    std_auc = model.avg_test_metrics.get('auc_std', 0)
                
                # Plot the mean TEST ROC curve with the correct AUC value
                plt.plot(mean_fpr, mean_tpr, color=color, lw=2,
                        label=f'{name} Test (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
                
                # Plot the standard deviation area
                plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                                color=color, alpha=0.2)
            
            # Calculate and plot the average TRAIN ROC curve if requested
            if include_train and train_tprs:
                # Interpolate all TPRs to the same FPR grid
                interp_train_tprs = []
                for fpr, tpr in zip(train_fprs, train_tprs):
                    interp_train_tprs.append(np.interp(mean_fpr, fpr, tpr))
                    interp_train_tprs[-1][0] = 0.0  # Force starting at (0,0)
                
                # Calculate mean TPR and standard deviation
                mean_train_tpr = np.mean(interp_train_tprs, axis=0)
                mean_train_tpr[-1] = 1.0  # Force ending at (1,1)
                
                # Calculate AUC of the mean curve
                mean_train_auc = auc(mean_fpr, mean_train_tpr)
                
                # Use the same AUC value that's in the model's metrics
                if 'auc' in model.avg_train_metrics:
                    mean_train_auc = model.avg_train_metrics['auc']
                
                # Plot the mean TRAIN ROC curve with the correct AUC value
                plt.plot(mean_fpr, mean_train_tpr, color=color, lw=2, linestyle='--',
                        label=f'{name} Train (AUC = {mean_train_auc:.3f})')
        else:
            # For single-run models, plot one ROC curve
            X_train, X_test, y_train, y_test = model.split_data(random_state=42)
            
            # Check if binary classification
            if len(np.unique(y_test)) == 2:
                # Get probability predictions
                if hasattr(model.model, "predict_proba"):
                    # Test data ROC
                    y_test_score = model.model.predict_proba(X_test)[:, 1]
                    test_fpr, test_tpr, _ = roc_curve(y_test, y_test_score)
                    test_roc_auc = auc(test_fpr, test_tpr)
                    
                    # Use the same AUC value that's in the model's metrics
                    if 'auc' in model.test_metrics:
                        test_roc_auc = model.test_metrics['auc']
                    
                    # Plot TEST ROC curve with the correct AUC value
                    plt.plot(test_fpr, test_tpr, color=color, lw=2,
                            label=f'{name} Test (AUC = {test_roc_auc:.3f})')
                    
                    # Training data ROC if requested
                    if include_train:
                        y_train_score = model.model.predict_proba(X_train)[:, 1]
                        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_score)
                        train_roc_auc = auc(train_fpr, train_tpr)
                        
                        # Use the same AUC value that's in the model's metrics
                        if 'auc' in model.train_metrics:
                            train_roc_auc = model.train_metrics['auc']
                        
                        # Plot TRAIN ROC curve with the correct AUC value
                        plt.plot(train_fpr, train_tpr, color=color, lw=2, linestyle='--',
                                label=f'{name} Train (AUC = {train_roc_auc:.3f})')
                else:
                    print(f"Model {name} doesn't support predict_proba, skipping")
            else:
                print(f"ROC curves are designed for binary classification")
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return plt.gcf()  # Return the current figure