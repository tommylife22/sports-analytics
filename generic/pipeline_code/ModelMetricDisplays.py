# ModelMetricDisplays.py
# Main module that imports and combines all metric display functionality

# Import all the sub-modules
from .ClassAnalysis import print_class_characteristics_analysis
from .ConfusionMatrixAnalysis import print_detailed_confusion_matrix_for_model
from .ModelComparison import compare_model_metrics

def display_predictions_vs_actuals(models):
    """
    Display predictions vs actual values side by side for all models.
    Shows the actual values in the first column and predictions from each model in separate columns.
    For classification problems, also shows detailed class analysis and model performance summary.
    
    Parameters:
    models : dict
        Dictionary containing model objects
    """
    import pandas as pd
    import numpy as np
    
    print("\nModel Predictions Side-by-Side Comparison:")
    print("=" * 60)
    
    # Check if we have any models to display
    if not models:
        print("No models to display predictions for.")
        return
    
    # Get a list of model names for our column headers
    model_names = list(models.keys())
    
    # Get test data from the first model (assuming all models use same test/train split)
    first_model_name = model_names[0]
    X_train, X_test, y_train, y_test = models[first_model_name].split_data(
        split_type="time",
        time_col="startDate"
    )
    
    # Create a DataFrame with actual values
    results = pd.DataFrame({
        'Actual': y_test
    })
    
    # Add predictions from each model
    predictions_dict = {}
    for name, model in models.items():
        # Get predictions for this model
        y_pred = model.model.predict(X_test)
        
        # Add to results DataFrame
        results[name] = y_pred
        predictions_dict[name] = y_pred
    
    # Display the combined results (ALWAYS show this)
    try:
        if pd.api.types.is_numeric_dtype(results['Actual']) and results['Actual'].max() > 1000:
            pd.set_option('display.float_format', '{:.2f}'.format)
            print(results.head(10))
            # Restore default display format
            pd.reset_option('display.float_format')
        else:
            print(results.head(10))
    except:
        # If any error with formatting, just print normally
        print(results.head(10))
    
    # Check for duplicate analysis prevention
    import sys
    current_frame = sys._getframe()
    caller_frame = current_frame.f_back
    
    # Get unique identifier for this call
    call_id = f"{id(models)}_{hash(tuple(models.keys()))}"
    
    # Prevent duplicate detailed analysis (but allow predictions table)
    if not hasattr(display_predictions_vs_actuals, '_analysis_calls'):
        display_predictions_vs_actuals._analysis_calls = set()
    
    # NEW ADDITION: For classification problems, show class analysis and model performance
    sample_model = models[first_model_name]
    if sample_model.problem_type == 'classification' and call_id not in display_predictions_vs_actuals._analysis_calls:
        display_predictions_vs_actuals._analysis_calls.add(call_id)
        
        # Show class characteristics analysis first
        print_class_characteristics_analysis(models)
        
        # Show simplified model performance with accuracy scores
        print(f"\n--- Model Performance ---")
        
        # Get accuracy scores for each model and sort by performance
        model_accuracy_scores = []
        for name, model in models.items():
            # Check if we have averaged metrics or regular metrics
            if hasattr(model, 'avg_test_metrics') and model.avg_test_metrics is not None:
                accuracy_score = model.avg_test_metrics.get('test_accuracy', 0) or model.avg_test_metrics.get('accuracy', 0)
            else:
                accuracy_score = model.test_metrics.get('test_accuracy', 0) or model.test_metrics.get('accuracy', 0)
            model_accuracy_scores.append((name, accuracy_score))
        
        # Sort by accuracy score (highest first)
        model_accuracy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create performance string
        performance_parts = [f"{name} Acc={score:.3f}" for name, score in model_accuracy_scores]
        performance_line = " > ".join(performance_parts)
        print(f"OVERALL: {performance_line}")
        
        # Show per-class accuracy for each model
        print(f"\nPER-CLASS ACCURACY:")
        
        # Get ALL unique classes from the full dataset (not just test set)
        all_classes_full = sorted(np.unique(np.concatenate([y_train, y_test])))
        classes_in_test = sorted(np.unique(y_test))
        
        # Calculate per-class accuracy for each model
        for name, predictions in predictions_dict.items():
            print(f"{name}:")
            
            for class_label in all_classes_full:
                # Find samples that are actually this class in test set
                class_mask = (y_test == class_label)
                total_class_samples = class_mask.sum()
                
                if total_class_samples > 0:  # If we have samples of this class in test set
                    # How many of this class did we get right?
                    class_predictions = predictions[class_mask]
                    correct_predictions = (class_predictions == class_label).sum()
                    class_acc = correct_predictions / total_class_samples
                    print(f"  Class {class_label}: {class_acc:.3f} ({correct_predictions}/{total_class_samples})")
                else:
                    # No samples of this class in test set
                    print(f"  Class {class_label}: N/A (0/0 - not in test set)")
        
    return results


# Optional: Function to show detailed confusion matrices if needed
def display_detailed_confusion_matrices(models):
    """
    Display detailed confusion matrix analysis for all models.
    Call this separately if you want the full detailed breakdown.
    
    Parameters:
    models : dict
        Dictionary containing model objects
    """
    # Get test data from the first model
    first_model_name = next(iter(models.keys()))
    X_train, X_test, y_train, y_test = models[first_model_name].split_data()
    
    # Get predictions from each model
    predictions_dict = {}
    for name, model in models.items():
        y_pred = model.model.predict(X_test)
        predictions_dict[name] = y_pred
    
    # Show detailed confusion matrices
    sample_model = models[first_model_name]
    if sample_model.problem_type == 'classification':
        print(f"\n{'='*80}")
        print("DETAILED CONFUSION MATRIX ANALYSIS FOR ALL MODELS")
        print(f"{'='*80}")
        
        # Show confusion matrix for each model
        for model_name, predictions in predictions_dict.items():
            print_detailed_confusion_matrix_for_model(model_name, y_test, predictions)