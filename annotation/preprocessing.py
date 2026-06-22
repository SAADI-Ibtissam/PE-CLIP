from glob import glob
import os

def update(file, old_str, new_str):
    print(f"Updating file: {file}")
    file_data = ""

    # Normalize both paths once so they match the normalized lines
    old_str = old_str.replace("\\", "/").rstrip("/")
    new_str = new_str.replace("\\", "/").rstrip("/") + "/"

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            print(f"Original line: {line.strip()}")
            # normalize the line to forward slashes for consistent matching
            normalized_line = line.replace("\\", "/")
            if old_str in normalized_line:
                updated_line = normalized_line.replace(old_str, new_str)
                print(f"Updated line: {updated_line.strip()}")
                line = updated_line
            else:
                # also show why it didn't match (once you’re confident, you can remove this)
                print(f"(no match for '{old_str}')")
            file_data += line

    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)

# --- YOUR CONFIG (same variables, just works now) ---
# --- Example of AFEW dataset --- #
old_path = "Code4/AFEWdataset/AFEW_Face_Retina"
new_path = "D:/KmuProj2/Code4/AFEWdataset/AFEW_Face_Retina"

# Find all text files (current folder). If needed, use recursive: glob('**/AFEW_*.txt', recursive=True)
all_txt_files = glob('AFEW_*.txt')
print("Found files:", all_txt_files)

for txt_file in all_txt_files:
    update(txt_file, old_path, new_path)
