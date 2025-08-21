import pandas as pd
import glob


csv_files_path = r'C:\\Users\\somlu\\OneDrive\\Desktop\\Polygence Project\\Data Collector\\*.csv'

# Use glob to get a list of all CSV files
csv_files = glob.glob(csv_files_path)

# Initialize an empty DataFrame to hold combined data
combined_data = pd.DataFrame()

# Iterate through each CSV file and concatenate into the combined_data DataFrame
for file in csv_files:
    data = pd.read_csv(file)
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Remove duplicate rows based on all columns
combined_data.drop_duplicates(inplace=True)

# Save the combined DataFrame to a new CSV file
combined_data.to_csv('combined_cas9_data.csv', index=False)

print("Combined CSV files saved as 'combined_cas9_data.csv'.")