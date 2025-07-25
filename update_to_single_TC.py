#!/usr/bin/env python3
"""
Script to update files with new base file data while preserving additional columns.
Usage: python update_to_single_TC.py folder_of_files_to_update suffix_of_files
Example: python update_to_single_TC.py t_data SST
"""

import sys
import os
from pathlib import Path

def update_files(folder_to_update, suffix):
    """
    Update files with new base file data.
    
    Args:
        folder_to_update: Folder containing files to update
        suffix: Suffix of files to update (e.g., 'SST')
    """
    base_folder = "./single_TC"
    update_path = Path(folder_to_update)
    base_path = Path(base_folder)
    
    if not update_path.exists():
        print(f"Error: Folder '{folder_to_update}' does not exist")
        return
    
    if not base_path.exists():
        print(f"Error: Base folder '{base_folder}' does not exist")
        return
    
    # Find all files to update
    files_to_update = list(update_path.glob(f"*_{suffix}.txt"))
    
    if not files_to_update:
        print(f"No files with suffix '_{suffix}.txt' found in {folder_to_update}")
        return
    
    print(f"Found {len(files_to_update)} files to update")
    
    for file_to_update in files_to_update:
        # Extract base filename by removing the suffix
        base_filename = file_to_update.name.replace(f"_{suffix}.txt", ".txt")
        base_file = base_path / base_filename
        
        if not base_file.exists():
            print(f"Warning: Base file '{base_file}' not found for '{file_to_update.name}', skipping...")
            continue
        
        print(f"Updating {file_to_update.name}...")
        
        try:
            # Read the base file
            with open(base_file, 'r') as f:
                base_lines = f.readlines()
            
            # Read the file to update
            with open(file_to_update, 'r') as f:
                file_lines = f.readlines()
            
            # Check if files have the same number of lines
            if len(base_lines) != len(file_lines):
                print(f"  Warning: Line count mismatch - base: {len(base_lines)}, current: {len(file_lines)}")
                print(f"  Using minimum line count: {min(len(base_lines), len(file_lines))}")
            
            # Create updated lines
            updated_lines = []
            num_lines = min(len(base_lines), len(file_lines))
            
            for i in range(num_lines):
                base_line = base_lines[i].rstrip('\n')
                file_line = file_lines[i].rstrip('\n')
                
                # Split the current file line to get base part and additional columns
                # The base part has 20 columns (ending with -999)
                file_parts = file_line.split(',')
                
                if len(file_parts) <= 21:
                    # No additional data in this line, just use base line
                    updated_lines.append(base_line + '\n')
                else:
                    additional_columns = ','.join(file_parts[21:])
                    updated_line = base_line + ',' + additional_columns + '\n'
                    updated_lines.append(updated_line)
            
            # Write the updated file
            with open(file_to_update, 'w') as f:
                f.writelines(updated_lines)
            
            print(f"  Successfully updated {file_to_update.name}")
            
        except Exception as e:
            print(f"  Error updating {file_to_update.name}: {str(e)}")
    
    print("\nUpdate complete!")

def main():
    if len(sys.argv) != 3:
        print("Usage: python update_to_single_TC.py folder_of_files_to_update suffix_of_files")
        print("Example: python update_to_single_TC.py t_data SST")
        sys.exit(1)
    
    folder_to_update = sys.argv[1]
    suffix = sys.argv[2]
    
    update_files(folder_to_update, suffix)

if __name__ == "__main__":
    main()
