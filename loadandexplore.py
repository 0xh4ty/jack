import pandas as pd

def load_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def load_excel(file_path, sheet_name=0):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    return data
  
def explore_data(data):
    print("\nFirst 5 Rows of the Dataset:")
    print(data.head())
    
    print("\nBasic Information:")
    print(data.info())
    
    print("\nSummary Statistics:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nColumn Data Types:")
    print(data.dtypes)

csv_file_path = 'path/to/your/csvfile.csv'
csv_data = load_csv(csv_file_path)

excel_file_path = 'path/to/your/excelfile.xlsx'
excel_data = load_excel(excel_file_path, sheet_name='Sheet1')

print("CSV Data Exploration:")
explore_data(csv_data)

print("\nExcel Data Exploration:")
explore_data(excel_data)
