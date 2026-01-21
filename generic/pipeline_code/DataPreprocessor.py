# DataPreprocessor

import pandas as pd
import numpy as np
import os
import sqlite3
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, file_name=None, target_variable=None):
        self.file_name = file_name
        self.target_variable = target_variable
        self.df = None
        self.scaler = None  # Add this to store the scaler
        self.columns_to_scale = []  # Add this to track columns for scaling
        if file_name:
            self.base_name = os.path.splitext(os.path.basename(file_name))[0]
        else:
            self.base_name = 'data'
    
    def load_data(self):
        # Load data from various file formats
        _, file_extension = os.path.splitext(self.file_name)
        file_extension = file_extension.lower()
        if file_extension == '.csv':
            data = pd.read_csv(self.file_name)
        elif file_extension in ['.xlsx', '.xls']:
            data = pd.read_excel(self.file_name)
        elif file_extension == '.json':
            data = pd.read_json(self.file_name)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Save original data for exploration
        globals()[f'df_original_{self.base_name}'] = data
        
        # Create a copy for processing
        self.df = data.copy()
        
        return self
    
    def explore_data(self):
        df = self.df
        
        print('\nDataset Overview:')
        print(f'Shape: {df.shape[0]} rows, {df.shape[1]} columns')
        
        print('\nColumn Types:')
        print(df.dtypes)
        
        print('\nMissing Values:')
        missing = df.isnull().sum()
        for col in missing[missing > 0].index:
            percent = (missing[col] / len(df)) * 100
            print(f"{col}: {missing[col]} missing values ({percent:.1f}%)")
            
        # Look at number columns and their values values
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            stats = df[col].describe()
            
            print(f'\nLooking at {col}:')
            print(f"Smallest value: {stats['min']}")
            print(f"Biggest value: {stats['max']}")
            print(f"Average value: {stats['mean']:.2f}")
            
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                zero_percent = (zero_count / len(df)) * 100
                print(f"Found {zero_count} zeros ({zero_percent:.1f}%)")
            
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"Found {neg_count} negative values")
        
        return self
    
    def handle_missing_values(self, nonsense_values=None):
        nonsense_values = nonsense_values or {}
        
        for column in self.df.columns:
            if column != self.target_variable:
                # Only process a column if it has missing values
                if self.df[column].isnull().any():
                    self._impute_values(column, np.nan)
                
                # Only process nonsense values if they were specifically provided
                if nonsense_values and column in nonsense_values:
                    self._impute_values(column, nonsense_values[column])
        
        return self
        
    def _impute_values(self, column_name, bad_value):
        # Check if the column contains non-numeric data
        if self.df[column_name].dtype == 'object':
            # Use empty string ('') as the fill value for categorical columns
            fill_value = ''
            num_missing = 0

            if pd.isna(bad_value):
                # Fill standard NaN values
                missing_mask = self.df[column_name].isnull()
                num_missing = missing_mask.sum()
                if num_missing > 0:
                    self.df.loc[missing_mask, column_name] = fill_value
            else:
                # Replace specific 'bad_value'
                missing_mask = (self.df[column_name] == bad_value)
                num_missing = missing_mask.sum()
                if num_missing > 0:
                    self.df.loc[missing_mask, column_name] = fill_value

            if num_missing > 0:
                print(f"Filled {num_missing} missing/bad values in object column '{column_name}' with empty string ('')")

        else: # Handle numeric imputation logic
            try:
                # Attempt conversion to numeric, coercing errors to NaN
                self.df[column_name] = pd.to_numeric(self.df[column_name], errors='coerce')

                numeric_bad_value = np.nan # Treat bad_value as NaN for consistency after coercion
                if not pd.isna(bad_value):
                    try:
                        # If bad_value itself was numeric, capture it
                        numeric_bad_value = float(bad_value)
                    except (ValueError, TypeError):
                        # If bad_value wasn't numeric, we treat it like other coerced NaNs
                        pass

                # Identify all values that are NaN (originally or after coercion)
                # or equal to a specific numeric bad_value
                initial_nan_mask = self.df[column_name].isnull()
                bad_value_mask = pd.Series([False] * len(self.df), index=self.df.index)
                if not pd.isna(numeric_bad_value):
                    # Ensure comparison is done safely with floating point numbers if necessary
                    if self.df[column_name].dtype == 'float':
                        bad_value_mask = np.isclose(self.df[column_name], numeric_bad_value)
                    else:
                        bad_value_mask = (self.df[column_name] == numeric_bad_value)


                # Combine masks: target NaNs and specific numeric bad values
                impute_mask = initial_nan_mask | bad_value_mask
                valid_mask = ~impute_mask

                num_to_impute = impute_mask.sum()

                if num_to_impute > 0:
                    if valid_mask.any():
                        mean_value = self.df.loc[valid_mask, column_name].mean()
                        std_value = self.df.loc[valid_mask, column_name].std()

                        # Handle cases where mean/std might be NaN/0
                        if pd.isna(mean_value): mean_value = 0
                        if pd.isna(std_value) or std_value <= 0: std_value = 1.0 # Avoid 0 or negative std dev

                        # Generate random values around the mean and std
                        random_values = np.random.normal(mean_value, std_value, size=num_to_impute)
                        self.df.loc[impute_mask, column_name] = random_values
                        print(f"Imputed {num_to_impute} missing/bad numeric values in '{column_name}' around mean {mean_value:.2f} (std: {std_value:.2f}).")
                    else:
                        # Fallback if ALL values were invalid/missing
                        fill_numeric = 0.0
                        self.df.loc[impute_mask, column_name] = fill_numeric
                        print(f"Filled all {num_to_impute} missing/bad numeric values in '{column_name}' with {fill_numeric} as fallback.")

            except Exception as e:
                print(f"Error processing numeric column {column_name}: {e}")
 
    def convert_dtypes(self):
        string_columns = self.df.select_dtypes(include="object").columns
        self.df[string_columns] = self.df[string_columns].astype(str)
        return self
    
    def one_hot_encode_binary_x(self):
        # Look at all columns except the target
        columns_to_check = [col for col in self.df.columns if col != self.target_variable]
        columns_to_drop = []
        new_columns = {}
        
        for col in columns_to_check:
            # Check if column has exactly 2 unique values
            if self.df[col].nunique() == 2:
                # Get unique values
                unique_vals = self.df[col].unique()
                # Create a single column for one of the values (true/false)
                new_col_name = f"{col}_{unique_vals[1]}"
                new_columns[new_col_name] = (self.df[col] == unique_vals[1]).astype(bool)
                columns_to_drop.append(col)
        
        if new_columns:
            # Combine all new columns at once
            new_df = pd.concat([self.df.drop(columns=columns_to_drop), pd.DataFrame(new_columns)], axis=1)
            self.df = new_df.copy()  # Create a fresh copy to defragment
        
        # Update the working data
        globals()[f'working_data_{self.base_name}'] = self.df
        return self
    
    def encode_categorical(self, categorical_cols=None, max_unique_values=100):
        """
        Encode categorical columns with one-hot encoding, but only if they have 
        fewer unique values than max_unique_values.
        """
        if categorical_cols is None:
            categorical_cols = self.df.select_dtypes(include='object').columns.tolist()
    
        if self.target_variable in categorical_cols:
            categorical_cols.remove(self.target_variable)
    
        columns_to_drop = []
        new_columns = {}
    
        # Print info about categorical columns
        print("\nCategorical Columns Analysis:")
        print("=" * 50)
    
        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            print(f"\nColumn: {col}")
            print(f"   Number of unique values: {n_unique}")
    
            if n_unique > 0 and n_unique <= max_unique_values:
                # Get unique values in their original order
                unique_vals = self.df[col].unique()
                print("   Status: Converting to binary columns")
                print(f"   Creating columns for: {', '.join(unique_vals[:-1])}")
                print(f"   Reference category (dropped): {unique_vals[-1]}")
    
                # Create new columns for this category (all except the last value)
                for val in unique_vals[:-1]:
                    new_colname = f"{col}_{val}"  # Added underscore between column name and value
                    new_columns[new_colname] = (self.df[col] == val).astype(bool)
                columns_to_drop.append(col)  # Only add the column once after processing all its values
                
            elif n_unique > max_unique_values:
                print(f"   Status: Skipped - too many unique values ({n_unique} > {max_unique_values})")
    
            print("-" * 50)
    
        # Only update dataframe if we have new columns
        if new_columns:
            # Combine all new columns at once
            new_df = pd.concat([self.df.drop(columns=columns_to_drop), pd.DataFrame(new_columns)], axis=1)
            self.df = new_df.copy()  # Create a fresh copy to defragment
    
        return self
    
    def encode_target(self):
        """
        Automatically detects and encodes target values for classification or regression.
        
        For datasets with 10 or fewer unique values, treats it as classification.
        For continuous or many-valued targets, treats it as regression.
        """
        # Get unique values
        unique_values = sorted(self.df[self.target_variable].unique())
        num_unique_values = len(unique_values)
        
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(self.df[self.target_variable]):
            # If already starts from 0 and is continuous, skip
            if min(unique_values) == 0 and max(unique_values) == num_unique_values - 1:
                print(f"\n{self.target_variable} is already correctly encoded - skipping!")
                return self
            
            # For 10 or fewer unique values, treat as classification
            if num_unique_values <= 10:
                print(f"\nDetected classification problem with {num_unique_values} unique values")
                
                # Create mapping starting from 0
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                
                # Apply the mapping
                self.df[self.target_variable] = self.df[self.target_variable].map(value_map)
                
                # Show what changed
                print(f"\nChanged {self.target_variable} values:")
                for old, new in value_map.items():
                    print(f"{old} → {new}")
                
                return self
            
            # If more than 10 unique values, print a note about regression
            print(f"\n{self.target_variable} has {num_unique_values} unique values - treating as regression")
        
        return self
        
    def scale_numerical(self):
        """Mark columns for scaling but don't actually scale yet"""
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        numerical_columns = [col for col in numerical_columns if col != self.target_variable]
        
        # Store the columns to scale but don't scale yet
        self.columns_to_scale = list(numerical_columns)
        print(f"Marked {len(self.columns_to_scale)} columns for scaling: {self.columns_to_scale}")
        
        return self
    
    def fit_scaler(self, data):
        """Fit the scaler on training data only"""
        if hasattr(self, 'columns_to_scale') and self.columns_to_scale:
            self.scaler = StandardScaler()
            self.scaler.fit(data[self.columns_to_scale])
            print()
        return self
    
    def transform_data(self, data):
        """Transform data using the fitted scaler"""
        if self.scaler and hasattr(self, 'columns_to_scale'):
            data_copy = data.copy()
            data_copy[self.columns_to_scale] = self.scaler.transform(data[self.columns_to_scale])
            return data_copy
        return data.copy()
    
    def move_target_to_end(self):
        cols = [col for col in self.df.columns if col != self.target_variable] + [self.target_variable]
        self.df = self.df[cols]
        return self
    
    def export_data(self, output_file=None):
        if output_file is None:
            base_name = os.path.splitext(self.file_name)[0]
            output_file = f"{base_name}_processed{os.path.splitext(self.file_name)[1]}"
            
        _, file_extension = os.path.splitext(output_file)
        file_extension = file_extension.lower()
        
        if file_extension == '.csv':
            self.df.to_csv(output_file, index=False)
        elif file_extension in ['.xlsx', '.xls']:
            self.df.to_excel(output_file, index=False)
        elif file_extension == '.json':
            self.df.to_json(output_file, orient='records')
        else:
            raise ValueError(f"Unsupported output file type: {file_extension}")
            
        # Save final processed data to a global variable
        globals()[f'processed_data_{self.base_name}'] = self.df
        print(f"\nFinal processed data saved as '{output_file}'")
        return self
    
def preprocess_data(target_column, file_path=None, df=None, skip_preprocessing=False):
    """
    Runs the full data preprocessing pipeline or simply loads the data.
    Returns the processed DataFrame and the preprocessor instance.
    """
    if file_path is None and df is None:
        raise ValueError("You must provide either 'file_path' or 'df'.")

    preprocessor = DataPreprocessor(file_path, target_column)

    if skip_preprocessing:
        print("\nSkipping preprocessing. Loading data directly.")
        if df is not None:
            clean_data = df.copy()
        else:
            preprocessor.load_data()
            clean_data = preprocessor.df.copy()
        return clean_data, None

    # --- Run the full pipeline ---
    print("\n=== Starting Data Processing Pipeline ===")
    
    if df is not None:
        preprocessor.df = df.copy()
    else:
        preprocessor.load_data()
    
    processed_df = (preprocessor
                    .explore_data()
                    # .handle_missing_values()
                    .convert_dtypes()
                    .encode_target()
                    .encode_categorical()
                    .scale_numerical()
                    .move_target_to_end()
                    .df)

    print("\n✅ Preprocessing pipeline complete.")
    return processed_df, preprocessor