# MissingValueHandler

import pandas as pd
import numpy as np

class MissingValueHandler:
    """
    A simple class to handle missing values in pandas DataFrames.
    """
    
    def __init__(self, df=None):
        """Initialize with an optional DataFrame"""
        self.df = df.copy() if df is not None else None
    
    def set_dataframe(self, df):
        """Set the DataFrame to process"""
        self.df = df.copy()
        return self
    
    def get_dataframe(self):
        """Return the processed DataFrame"""
        return self.df
    
    def fill_column(self, column, value):
        """
        Fill missing values in a specific column with a given value
        
        Parameters:
        -----------
        column : str
            Column name to process
        value : value or str
            Value to fill with. Can be a constant or one of:
            - 'mean': Fill with random values around the mean (numeric only)
            - 'median': Fill with random values around the median (numeric only)
            - 'zero': Fill with 0
            - 'empty': Fill with empty string (for object columns)
        """
        if self.df is None:
            print("No DataFrame has been set")
            return self
            
        if column not in self.df.columns:
            print(f"Column '{column}' not found in DataFrame")
            return self
            
        missing_count = self.df[column].isnull().sum()
        if missing_count == 0:
            return self  # Skip if no missing values to fill
            
        # Handle statistical methods
        if value == 'mean' and pd.api.types.is_numeric_dtype(self.df[column]):
            mean_value = self.df[column].mean()
            std_value = self.df[column].std() 
            
            # Handle case where std is 0 or NA
            if pd.isna(std_value) or std_value == 0:
                std_value = mean_value * 0.1 if mean_value != 0 else 1.0
                
            random_values = np.random.normal(mean_value, std_value, size=missing_count)
            
            # Locate missing values and fill them
            self.df.loc[self.df[column].isnull(), column] = random_values
            print(f"Filled {missing_count} NaN values in '{column}' with random values around mean: {mean_value:.2f} (std: {std_value:.2f})")
            
        elif value == 'median' and pd.api.types.is_numeric_dtype(self.df[column]):
            median_value = self.df[column].median()
            std_value = self.df[column].std()
            
            # Handle case where std is 0 or NA
            if pd.isna(std_value) or std_value == 0:
                std_value = median_value * 0.1 if median_value != 0 else 1.0
                
            random_values = np.random.normal(median_value, std_value, size=missing_count)
            
            # Locate missing values and fill them
            self.df.loc[self.df[column].isnull(), column] = random_values
            print(f"Filled {missing_count} NaN values in '{column}' with random values around median: {median_value:.2f} (std: {std_value:.2f})")
            
        elif value == 'zero':
            self.df[column] = self.df[column].fillna(0)
            print(f"Filled {missing_count} NaN values in '{column}' with 0")
            
        elif value == 'empty' and pd.api.types.is_object_dtype(self.df[column]):
            self.df[column] = self.df[column].fillna('')
            print(f"Filled {missing_count} NaN values in '{column}' with empty string")
            
        else:
            # Use the provided value directly
            self.df[column] = self.df[column].fillna(value)
            print(f"Filled {missing_count} NaN values in '{column}' with: {value}")
            
        return self
    
    def fill_columns(self, fill_dict):
        if self.df is None:
            print("No DataFrame has been set")
            return self
        
        if fill_dict:
            print("\nFilling missing values:")
            for column, value in fill_dict.items():
                self.fill_column(column, value)
        else:
            print("\nNo fill strategy provided, keeping missing values as-is")
            
        return self
    
    def drop_rows_with_missing(self):
        """Drop all rows that contain any missing values"""
        if self.df is None:
            print("No DataFrame has been set")
            return self
            
        old_count = len(self.df)
        self.df = self.df.dropna()
        new_count = len(self.df)
        dropped = old_count - new_count
        
        print(f"Dropped {dropped} rows with missing values ({dropped/old_count*100:.1f}% of data)")
        return self