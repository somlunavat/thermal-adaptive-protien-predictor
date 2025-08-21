from Bio.Blast import NCBIWWW, NCBIXML
import pandas as pd
import time

# Function to use BLAST on a sequence and return results
def run_blast(sequence, hitlist_size=5000):
    try:
        print(f"doing BLAST for : {sequence[:30]}...")
        result_handle = NCBIWWW.qblast("blastp", "nr", sequence, hitlist_size=hitlist_size)
        blast_record = NCBIXML.read(result_handle)

        #get values and append to results
        results = []
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                if hsp.expect < 0.05:  # Filter based on E-value threshold
                    description = alignment.title
                    scientific_name = alignment.hit_def.split("[")[-1].split("]")[0]  # gets the scientific name
                    query_cover = (hsp.query_end - hsp.query_start + 1) / len(sequence) * 100  # getting query percent
                    e_value = hsp.expect
                    identity_percentage = (hsp.identities / hsp.align_length) * 100
                    bit_score = hsp.bits
                    accession = alignment.accession
                    alignment_length = hsp.align_length
                    start_end_positions_query = f"{hsp.query_start}-{hsp.query_end}"
                    start_end_positions_subject = f"{hsp.sbjct_start}-{hsp.sbjct_end}"
                    mismatch_count = hsp.align_length - hsp.identities
                    gap_count = hsp.gaps

                    results.append({
                        'Description': description,
                        'Scientific Name': scientific_name,
                        'Query Cover (%)': query_cover,
                        'E-value': e_value,
                        'Identity (%)': identity_percentage,
                        'Bit Score': bit_score,
                        'Accession Number': accession,
                        'Alignment Length': alignment_length,
                        'Start-End Query': start_end_positions_query,
                        'Start-End Subject': start_end_positions_subject,
                        'Mismatch Count': mismatch_count,
                        'Gap Count': gap_count
                    })
        return results

    except Exception as e:
        print(f"error on BLAST: {e}")
        return None

# Function to process sequences, retrive data and store on csv
def process_sequences(input_file, output_file, start_row, end_row):
    try:
        df = pd.read_excel(input_file, sheet_name='Sequences')

        subset_df = df.iloc[start_row:end_row]
        sequences = subset_df['Full sequence'].dropna().unique()  # Drop missing and duplicate sequences

        all_results = []

        for i, sequence in enumerate(sequences, start=start_row+1):
            if pd.isna(sequence) or len(sequence.strip()) == 0:
                print(f"skipping row {i} cuz missing data.")
                continue

            print(f"doing sequence {i}: {sequence[:30]}...")  # know what sequence its on

            # Run BLAST
            results = run_blast(sequence)
            if results:
                all_results.extend(results)

            # Rate limiting (says 10 seconds on website)
            time.sleep(10)

        # Save results to a new CSV file
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(output_file, index=False)

        print(f"results on {output_file}")

    except Exception as e:
        print(f"error: {e}")


# Input and output files
input_file = 'CAS9_DATA.xlsx'
output_file = 'blast_results_part8.csv' #manually input numbers ;(

#start and ending row (cuz i cant do all at once D; )
start_row = 8512
end_row = 8562

process_sequences(input_file, output_file, start_row, end_row)
