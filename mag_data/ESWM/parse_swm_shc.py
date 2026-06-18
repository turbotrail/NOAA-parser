# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///

import os
import glob
import csv

def parse_shc_to_csv(shc_filepath):
    csv_filepath = shc_filepath.replace('.shc', '.csv')
    metadata = {}
    
    data_count = 0
    with open(shc_filepath, 'r') as f_in, open(csv_filepath, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['n', 'm', 'coefficient'])
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            
            parts = line.split()
            # Determine if it's a data line: n, m, coef
            # We assume n and m are integers, so parts[0] won't have a decimal point
            if len(parts) >= 3 and '.' not in parts[0]:
                writer.writerow(parts[:3])
                data_count += 1
            elif len(parts) >= 4 and '.' not in parts[0]:
                metadata['n_min'] = parts[0]
                metadata['n_max'] = parts[1]
            elif len(parts) == 1 and '.' in parts[0]:
                metadata['epoch'] = parts[0]
                
    print(f"File: {os.path.basename(shc_filepath)}")
    print(f"  Metadata: {metadata}")
    print(f"  Number of coefficients extracted: {data_count}")
    print(f"  Saved to {os.path.relpath(csv_filepath, '.')}\n")

if __name__ == "__main__":
    # Base directory for the ESWM data (current directory since script is now in ESWM)
    base_dir = "."
    shc_files = glob.glob(os.path.join(base_dir, "**/*.shc"), recursive=True)
    
    if not shc_files:
        print(f"No .shc files found in {os.path.abspath(base_dir)}")
    else:
        print(f"Found {len(shc_files)} .shc files. Parsing...\n")
        for f in shc_files:
            parse_shc_to_csv(f)
