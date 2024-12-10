import pandas as pd
import numpy as np
import re
import os

# Function to process each CSV file
def process_csv(file_path):
    # Step 1: Load the CSV file containing the string representation of the array
    df = pd.read_csv(file_path, header=None)  # Read CSV without headers

    # Step 2: Define a function to clean up the string representation and convert it to a NumPy array
    def parse_array(arr_str):
        # Check if the value is a string
        if isinstance(arr_str, str):
            # Step 2a: Remove the surrounding quotes
            arr_str = arr_str.strip().strip('"')

            # Step 2b: Replace the newlines and excessive spaces with a single space
            arr_str = arr_str.replace('\n', ' ').replace('  ', ' ')

            # Step 2c: Remove the square brackets to match the NumPy-style array format
            arr_str = re.sub(r'\[|\]', '', arr_str)

            # Step 2d: Convert the cleaned string into a list of floats and reshape into a 2D array (6 elements per row)
            arr_list = np.array([float(x) for x in arr_str.split()]).reshape(-1, 6)
        else:
            # If it's not a string, return an empty array (or handle as appropriate)
            arr_list = np.empty((0, 6))  # Modify this as needed to handle non-string values

        # Return the cleaned and reshaped array
        return arr_list

    # Step 3: Apply the parsing function to each row in the dataframe
    df_parsed = df[0].apply(parse_array)  # Apply the parse_array function to each element in the CSV

    # Step 4: Flatten the list of arrays into a single 2D array and return it as a DataFrame
    expanded_data = pd.DataFrame(np.vstack(df_parsed.tolist()))  # Combine arrays into one DataFrame

    return expanded_data

# Directory containing the CSV files
csv_directory = '/home/greenbaum-gpu/Reuben/keras-retinanet/output/csv_output'  # Replace with the path to your directory

# List to store the data from all CSV files
all_data = []

# Step 8: Loop through each file in the directory and process it
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):  # Process only CSV files
        file_path = os.path.join(csv_directory, filename)
        processed_data = process_csv(file_path)  # Process the CSV file
        all_data.append(processed_data)  # Append the processed data to the list

# Step 9: Combine all the data from all CSV files into one DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Directory where the final combined CSV will be saved
target_directory = '/home/greenbaum-gpu/Reuben/keras-retinanet/finaloutput'  # Replace with your desired target directory

# Step 10: Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

# Step 11: Construct the path for the final combined CSV file
final_file_path = os.path.join(target_directory, 'annotations.csv')

# Step 12: Save the combined data to a single CSV file without headers
combined_data.to_csv(final_file_path, index=False, header=False)  # Save as a single CSV without headers
