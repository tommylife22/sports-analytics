# ClassAnalysis.py
# Module for analyzing class characteristics in classification problems

def print_class_characteristics_analysis(models):
    """
    Print cluster-style analysis for classification problems showing
    average feature values for each class (like cluster analysis but for classes).
    
    Parameters:
    models : dict
        Dictionary containing model objects
    """
    import pandas as pd
    import numpy as np
    
    # Get the first model to check if it's classification
    sample_model = next(iter(models.values()))
    if sample_model.problem_type != 'classification':
        return  # Only run for classification problems
    
    # Get the data and target from the first model
    X_train, X_test, y_train, y_test = sample_model.split_data()
    
    # Combine train and test data to analyze all available data
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    
    # Create a DataFrame with features and target
    analysis_df = X_full.copy()
    analysis_df['actual_class'] = y_full
    
    print(f"\n{'='*80}")
    print("CLASS CHARACTERISTICS ANALYSIS")
    print(f"{'='*80}")
    print("Understanding what makes each class different by looking at average feature values")
    
    # Class distribution (like cluster distribution)
    print(f"\n--- Class Distribution ---")
    print("Number of samples in each class:")
    class_counts = analysis_df['actual_class'].value_counts().sort_index().to_frame()
    class_counts.columns = ['Count']
    class_counts['Percentage'] = (class_counts['Count'] / len(analysis_df) * 100).round(1)
    print(class_counts)
    
    # Class characteristics (like cluster characteristics)
    print(f"\n--- Class Characteristics ---")
    print("Average feature values for each class:")
    class_summary = analysis_df.groupby('actual_class').mean(numeric_only=True)
    
    # Round to 3 decimal places for readability
    class_summary = class_summary.round(3)
    print(class_summary)
    
    # Find the most distinguishing features for each class
    print(f"\n--- Standout Features ---")
    
    # Calculate overall means
    overall_means = analysis_df.select_dtypes(include=[np.number]).mean()
    
    for class_name in sorted(analysis_df['actual_class'].unique()):
        class_means = class_summary.loc[class_name]
        
        # Calculate how much each feature deviates from overall average
        deviations = {}
        for feature in class_means.index:
            if feature in overall_means.index:
                deviation = abs(class_means[feature] - overall_means[feature])
                # Calculate as percentage of overall mean to normalize
                if overall_means[feature] != 0:
                    pct_deviation = (deviation / abs(overall_means[feature])) * 100
                else:
                    pct_deviation = deviation * 100  # If overall mean is 0
                deviations[feature] = pct_deviation
        
        # Get the top distinguishing feature
        top_feature = max(deviations.items(), key=lambda x: x[1])
        feature_name, pct_dev = top_feature
        class_val = class_means[feature_name]
        overall_val = overall_means[feature_name]
        direction = "above" if class_val > overall_val else "below"
        
        print(f"Class {class_name}: {feature_name}={class_val:.2f} ({pct_dev:.0f}% {direction} average)")