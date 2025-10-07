import re
from pathlib import Path
import shutil

path = Path("MHW_detect.py")
backup = path.with_suffix(path.suffix + ".bak")
shutil.copy2(path, backup)

# Read the original file
with open('MHW_detect.py', 'r') as f:
    content = f.read()

# Check if pickle import exists, if not add it at the top
if 'import pickle' not in content:
    # Find the last import statement
    import_pattern = r'^import .*$|^from .* import .*$'
    imports = list(re.finditer(import_pattern, content, re.MULTILINE))
    
    if imports:
        # Insert after the last import
        last_import_end = imports[-1].end()
        content = content[:last_import_end] + '\nimport pickle' + content[last_import_end:]
    else:
        # No imports found, add at the beginning
        content = 'import pickle\n' + content

def replace_func(match):
    indent = match.group(1)
    original_line = match.group(2)
    fig_name = match.group(3)
    
    # Create the new lines to insert
    pkl_line = f'{indent}output_file = "mhw_plot_pkl/{fig_name}.pkl"'
    open_line = f'{indent}with open(output_file, \'wb\') as f:'
    dump_line = f'{indent}    pickle.dump(fig, f)'
    
    # Return the new lines followed by the original line
    return f'{pkl_line}\n{open_line}\n{dump_line}\n{indent}{original_line}'

# Replace all occurrences
pattern = r'([ \t]*)(output_file = "mhw_plot/(Fig\d+)\.pdf")'
modified_content = re.sub(pattern, replace_func, content)
pattern = r'([ \t]*)(output_file = "mhw_plot/(FigS\d+)\.pdf")'
modified_content = re.sub(pattern, replace_func, modified_content)

# Write the modified content back to the file
with open('MHW_detect.py', 'w') as f:
    f.write(modified_content)

print("MHW_detect.py has been successfully modified!")
print("Added pickle saving code above each PDF output statement.")
