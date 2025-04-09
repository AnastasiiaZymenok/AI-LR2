
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read the data
def read_data():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    data = pd.read_csv('income_data.txt', names=columns, delimiter=', ')
    return data

def analyze_data(data):
    # Basic information
    print("\nDataset Shape:", data.shape)
    print("\nFeature Names:", data.columns.tolist())
    
    # Display first few rows
    print("\nFirst few rows of data:")
    print(data.head())
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(data.describe())
    
    # Count of income categories
    print("\nIncome Distribution:")
    print(data['income'].value_counts())
    
    # Average age by income
    print("\nAverage age by income category:")
    print(data.groupby('income')['age'].mean())
    
    # Education distribution
    print("\nEducation Distribution:")
    print(data['education'].value_counts())

def main():
    # Read data
    data = read_data()
    
    # Clean data (remove '?' values)
    data = data.replace('?', np.nan)
    data = data.dropna()
    
    # Analyze data
    analyze_data(data)

if __name__ == "__main__":
    main()
