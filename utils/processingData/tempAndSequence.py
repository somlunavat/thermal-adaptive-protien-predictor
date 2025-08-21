from Bio import Entrez, SeqIO
import pandas as pd
import re

# Set your email for NCBI requests
Entrez.email = "somlunavat@gmail.com"

# Load the organism temperature database
temperature_df = pd.read_csv('all_species_temperatures.csv')  # Replace with the actual path to your temperature database CSV

# Function to fetch organism temperature using the local database
def fetch_organism_temperature(organism_name):
    try:
        # Find the temperature for the given organism name in the local database
        temperature_data = temperature_df.loc[
            temperature_df['Organism'] == organism_name, 'Consensus optimal temperature'].values
        if len(temperature_data) > 0:
            return temperature_data[0]
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error fetching temperature for {organism_name}: {e}")
        return "Unknown"

# Function to fetch the full protein sequence from NCBI
def fetch_protein_sequence(accession_number, row_index):
    try:
        print(f"Fetching protein sequence for row {row_index + 1}...")  # Row tracking for protein fetching
        handle = Entrez.efetch(db="protein", id=accession_number, rettype="fasta", retmode="text")
        sequence_record = SeqIO.read(handle, "fasta")
        handle.close()
        return str(sequence_record.seq)
    except Exception as e:
        print(f"Error retrieving sequence for {accession_number} at row {row_index + 1}: {e}")
        return None

# Function to extract the scientific name after "Cas9" in brackets
def extract_organism_name_from_description(description):
    try:
        # Use regex to find the organism name after "Cas9" in brackets
        match = re.search(r"Cas9 \[([^\]]+)\]", description)
        if match:
            return match.group(1).strip()
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error extracting organism name from description: {description}: {e}")
        return "Unknown"

# Function to extract accession number from the description
def extract_accession(description):
    try:
        # Safely extract the accession number
        return description.split('|')[1].strip()
    except IndexError:
        print(f"Error extracting accession from: {description}")
        return None

# Main function to process the input CSV file for a specific row range
def process_cas9_data(input_file, output_file, start_row=0, end_row=None):
    # Load the input CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Limit to the specified rows
    df = df[start_row:end_row]

    # Make new columns
    df['Organism Temperature'] = None
    df['Full Sequence'] = None

    if 'Description' in df.columns:
        for index, row in df.iterrows():
            # Try using description first
            organism_name = extract_organism_name_from_description(row['Description'])

            # If the first method fails, try using 'Scientific Name' column
            if organism_name == "Unknown" and 'Scientific Name' in df.columns:
                organism_name = row['Scientific Name']

            # Fetch the organism temperature using the determined organism name
            df.at[index, 'Organism Temperature'] = fetch_organism_temperature(organism_name)

    else:
        print("Column 'Description' not found in the input file.")

    # Fetch full protein sequences from NCBI
    if 'Description' in df.columns:
        for index, row in df.iterrows():
            accession_number = extract_accession(row['Description'])
            if accession_number:
                df.at[index, 'Full Sequence'] = fetch_protein_sequence(accession_number, index)
    else:
        print("Column 'Description' not found in the input file.")

    # Save the DataFrame to a new CSV
    df.to_csv(output_file, index=False)
    print(f"Updated data saved to {output_file}")

# Example usage
input_file = 'blast_results_part8.csv'
output_file = 'train4.csv'

# Process rows from 40,001 to 80,000
process_cas9_data(input_file, output_file, start_row=90000, end_row=100000)
