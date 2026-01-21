# ConfusionMatrixAnalysis.py
# Module for detailed confusion matrix analysis

def print_detailed_confusion_matrix_for_model(model_name, actual_values, predictions):
    """
    Print a detailed confusion matrix analysis for a single model
    in beginner-friendly language.
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd
    
    # Get unique classes
    classes = sorted(np.unique(np.concatenate([actual_values, predictions])))
    
    # Only proceed if we have more than 1 class
    if len(classes) < 2:
        print(f"\nSkipping confusion matrix for {model_name} - only found {len(classes)} class: {classes}")
        print("(Confusion matrix requires at least 2 classes to compare)")
        return
    
    # Create confusion matrix
    cm = confusion_matrix(actual_values, predictions, labels=classes)
    
    print(f"\n{'='*80}")
    print(f"DETAILED CONFUSION MATRIX FOR {model_name.upper()}")
    print(f"{'='*80}")
    
    print("\nWhat this tells us:")
    print("- TP (True Positive): Model correctly predicted this class")
    print("- FP (False Positive): Model incorrectly predicted this class") 
    print("- FN (False Negative): Model missed this class (said it was something else)")
    print("- TN (True Negative): Model correctly said it was NOT this class")
    
    # For each class, calculate TP, FP, FN, TN
    for i, class_name in enumerate(classes):
        print(f"\nCLASS '{class_name}' ANALYSIS:")
        print("-" * 40)
        
        # True Positives: diagonal element
        tp = cm[i, i]
        
        # False Positives: sum of column i, minus the diagonal
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        # False Negatives: sum of row i, minus the diagonal  
        fn = np.sum(cm[i, :]) - cm[i, i]
        
        # True Negatives: total - tp - fp - fn
        tn = np.sum(cm) - tp - fp - fn
        
        print(f"True Positives (TP):  {tp:4d} - Times model correctly said '{class_name}'")
        print(f"False Positives (FP): {fp:4d} - Times model wrongly said '{class_name}' (but it wasn't)")
        print(f"False Negatives (FN): {fn:4d} - Times model missed '{class_name}' (said something else)")
        print(f"True Negatives (TN):  {tn:4d} - Times model correctly said 'NOT {class_name}'")
        
        # Calculate simple metrics for this class
        if tp + fn > 0:  # Avoid division by zero
            recall = tp / (tp + fn)
            print(f"Recall (Sensitivity):  {recall:.3f} - How well we catch '{class_name}' when it's there")
        
        if tp + fp > 0:  # Avoid division by zero
            precision = tp / (tp + fp)
            print(f"Precision:             {precision:.3f} - When we say '{class_name}', how often we're right")
    
    print(f"\nCONFUSION MATRIX TABLE FOR {model_name.upper()}:")
    print("-" * 50)
    print("Rows = Actual/True class")
    print("Columns = Predicted class")
    print("Numbers = How many times this happened\n")
    
    # Create a nice looking confusion matrix table
    cm_df = pd.DataFrame(cm, 
                        index=[f"Actually_{cls}" for cls in classes],
                        columns=[f"Predicted_{cls}" for cls in classes])
    print(cm_df)
    
    # Find the most confused classes for this model
    print(f"\nMost Common Mistakes for {model_name}:")
    mistakes = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                mistakes.append((classes[i], classes[j], cm[i, j]))
    
    # Sort by frequency and show top 3
    mistakes.sort(key=lambda x: x[2], reverse=True)
    for actual, predicted, count in mistakes[:3]:  # Show top 3 to keep it concise
        print(f"- {count} times: Actually '{actual}' but predicted '{predicted}'")