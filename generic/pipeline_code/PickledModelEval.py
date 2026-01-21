# PickledModelEval

import pandas as pd
import numpy as np
from PickleModels import load_model
from DataPreprocessor import DataPreprocessor

def evaluate_model(model_path, test_file_path=None, test_df=None, target_column=None, output_file=None, skip_preprocessing=False):
    """
    Evaluate a pickled model against a test dataset
    
    Parameters:
    -----------
    model_path : str
        Path to the pickled model file
    test_file_path : str, optional
        Path to the test data file (use either this OR test_df)
    test_df : pandas.DataFrame, optional
        DataFrame containing test data (use either this OR test_file_path)
    target_column : str, optional
        Name of the target column in the test data (if available)
    output_file : str, optional
        Name of the file to save prediction results (default depends on if target exists)
    skip_preprocessing : bool, optional
        Whether to skip preprocessing (if your test data is already preprocessed)
        
    Returns:
    --------
    DataFrame with predictions (and comparison with actuals if target_column is provided)
    """
    # Validate inputs - need either file path or DataFrame
    if test_file_path is None and test_df is None:
        raise ValueError("You must provide either test_file_path or test_df")
    if test_file_path is not None and test_df is not None:
        raise ValueError("Provide either test_file_path or test_df, not both")
    
    # 1. Load the model
    print(f"Loading model from {model_path}")
    Model = load_model(model_path)

    # 2. Load and preprocess the test data
    if test_file_path is not None:
        print(f"Loading test data from {test_file_path}")
        preprocessor = DataPreprocessor(test_file_path, target_column)
        preprocessor.load_data()
    else:
        print(f"Using provided test DataFrame with shape {test_df.shape}")
        preprocessor = DataPreprocessor(None, target_column)
        preprocessor.df = test_df.copy()  # Use the provided DataFrame

    # Save the original test data (may or may not have actual values)
    original_test_data = preprocessor.df.copy()

    if not skip_preprocessing:
        # Apply same preprocessing steps
        print("Preprocessing test data...")
        clean_data = (preprocessor
                    .handle_missing_values()
                    .convert_dtypes()
                    .one_hot_encode_binary_x()
                    .encode_categorical()
                    .scale_numerical()
                    .df)
    else:
        print("Skipping preprocessing as requested...")
        clean_data = preprocessor.df

    # IMPORTANT: Get the exact columns that were used during training
    training_columns = Model.df.drop(columns=[Model.target_variable]).columns
    print(f"Number of features in training data: {len(training_columns)}")

    # Check for differences in columns between train and test data
    test_feature_columns = clean_data.columns
    if target_column:
        test_feature_columns = [col for col in test_feature_columns if col != target_column]
    print(f"Number of features in test data: {len(test_feature_columns)}")

    # Find missing columns (in train but not in test)
    missing_columns = set(training_columns) - set(test_feature_columns)
    if missing_columns:
        print(f"Missing columns in test data: {missing_columns}")
        # Add missing columns with zeros
        for col in missing_columns:
            clean_data[col] = 0

    # Find extra columns (in test but not in train)
    extra_columns = set(test_feature_columns) - set(training_columns)
    if extra_columns:
        print(f"Extra columns in test data (will be removed): {extra_columns}")
        # Remove extra columns (but don't remove target if it exists)
        if target_column and target_column in extra_columns:
            extra_columns.remove(target_column)
        clean_data = clean_data.drop(columns=list(extra_columns))

    # Make sure columns are in the same order as during training
    # And exclude the target column for prediction
    if target_column:
        X_test = clean_data.drop(columns=[target_column])
    else:
        X_test = clean_data
    
    # If the model has a preprocessor with a scaler, we need to use it to transform the data
    if hasattr(Model, 'preprocessor') and Model.preprocessor is not None:
        if hasattr(Model.preprocessor, 'scaler') and Model.preprocessor.scaler is not None:
            print("Using model's stored scaler to transform test data...")
            X_test = Model.preprocessor.transform_data(X_test)
    
    X_test = X_test[training_columns]
    print(f"Final feature count after alignment: {len(X_test.columns)}")

    # 3. Make predictions
    print("Making predictions...")
    predictions = Model.model.predict(X_test)

    # 4. Create a results DataFrame
    if target_column and target_column in original_test_data.columns:
        # We have actual values to compare against
        actual_values = original_test_data[target_column]
        results = pd.DataFrame({
            'Actual': actual_values,
            'Predicted': predictions
        })
        
        # Calculate metrics based on model type
        print("\n=== MODEL PERFORMANCE METRICS ===")
        if Model.problem_type == 'regression':
            # Add error columns
            results['Error'] = results['Predicted'] - results['Actual']
            results['Abs_Error'] = abs(results['Error'])
            results['Squared_Error'] = results['Error'] ** 2
            
            # Import metrics if we need them
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Calculate regression metrics
            mse = mean_squared_error(actual_values, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_values, predictions)
            r2 = r2_score(actual_values, predictions)
            
            # Additional regression metrics
            median_abs_error = np.median(np.abs(predictions - actual_values))
            
            # MAPE calculation (handles zeros appropriately)
            non_zero_mask = actual_values != 0
            if non_zero_mask.any():
                mape = np.mean(np.abs((actual_values[non_zero_mask] - predictions[non_zero_mask]) / 
                                      actual_values[non_zero_mask])) * 100
            else:
                mape = "Cannot calculate (all actual values are zero)"
            
            print("\nRegression Performance Metrics:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Median Absolute Error: {median_abs_error:.4f}")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape}")
            
            # Find worst predictions
            if non_zero_mask.any():
                results['Abs_Percent_Error'] = 0
                results.loc[non_zero_mask, 'Abs_Percent_Error'] = np.abs(
                    (results.loc[non_zero_mask, 'Actual'] - results.loc[non_zero_mask, 'Predicted']) / 
                    results.loc[non_zero_mask, 'Actual']) * 100
            
            print("\nWorst 5 Predictions (by Absolute Error):")
            print(results.nlargest(5, 'Abs_Error')[['Actual', 'Predicted', 'Error', 'Abs_Error']])
            
        else:
            # Import metrics for classification
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Classification metrics
            accuracy = accuracy_score(actual_values, predictions)
            
            # Check if binary or multiclass
            unique_classes = len(np.unique(actual_values))
            if unique_classes == 2:
                precision = precision_score(actual_values, predictions)
                recall = recall_score(actual_values, predictions)
                f1 = f1_score(actual_values, predictions)
                
                print("\nBinary Classification Metrics:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                
                # Calculate AUC if model supports predict_proba
                if hasattr(Model.model, "predict_proba"):
                    from sklearn.metrics import roc_curve, auc
                    import matplotlib.pyplot as plt
                    
                    # Get probability predictions
                    y_proba = Model.model.predict_proba(X_test)[:, 1]
                    
                    # Calculate ROC curve and AUC
                    fpr, tpr, _ = roc_curve(actual_values, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    print(f"AUC Score: {roc_auc:.4f}")
                    
                    # Create ROC curve plot
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc="lower right")
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Save the plot
                    plt.savefig("roc_curve.png")
                    print("ROC curve saved to roc_curve.png")
                
            else:
                # Multiclass
                precision = precision_score(actual_values, predictions, average='weighted')
                recall = recall_score(actual_values, predictions, average='weighted')
                f1 = f1_score(actual_values, predictions, average='weighted')
                
                print("\nMulticlass Classification Metrics:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Weighted Precision: {precision:.4f}")
                print(f"Weighted Recall: {recall:.4f}")
                print(f"Weighted F1 Score: {f1:.4f}")
        
        # Set default output file for comparison
        if output_file is None:
            output_file = "predictions_vs_actuals.csv"
    else:
        # No target column - just return predictions
        # Add an ID column from original data if it exists
        id_column = None
        for col_name in ['id', 'ID', 'Id']:
            if col_name in original_test_data.columns:
                id_column = col_name
                break
        
        if id_column:
            results = pd.DataFrame({
                id_column: original_test_data[id_column],
                'Predicted': predictions
            })
        else:
            # Use index as ID
            results = pd.DataFrame({
                'ID': original_test_data.index,
                'Predicted': predictions
            })
        
        print("\nNo target column provided - evaluation metrics not available")
        
        # Set default output file for just predictions
        if output_file is None:
            output_file = "predictions.csv"

    # 5. Save the results
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    # 6. Display sample results
    if target_column and target_column in original_test_data.columns:
        print("\nSample comparison (Predictions vs Actuals):")
    else:
        print("\nSample predictions:")
    print(results.head(10))
    
    return results