from pathlib import Path
import shutil

path = Path("MHW_detect.py")
backup = path.with_suffix(path.suffix + ".bak")
shutil.copy2(path, backup)

above = ["    if basin_code == 'AL':", "        basin_code = 'NA'"]
below = ["    if basin_code == 'NA':", "        basin_code = 'AL'"]
needle = "ax.set_title(f'{basin_code}:"

out = []
for line in path.read_text(encoding="utf-8").splitlines():
    if needle in line:
        out.extend(above)
        out.append(line)
        out.extend(below)
    else:
        out.append(line)

path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"Done. Backup written to {backup}")
