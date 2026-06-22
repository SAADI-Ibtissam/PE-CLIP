# from glob import glob
# import os

# def update(file, old_str, new_str):
#     print(f"Updating file: {file}")
#     file_data = ""
#     with open(file, "r", encoding="utf-8") as f:
#         for line in f:
#             print(f"Original line: {line.strip()}")
#             normalized_line = line.replace('\\', '/')
#             if old_str in normalized_line:
#                 updated_line = normalized_line.replace(old_str, new_str)
#                 print(f"Updated line: {updated_line.strip()}")
#                 line = updated_line
#             file_data += line
#     with open(file, "w", encoding="utf-8") as f:
#         f.write(file_data)

# # Example path updates based on your current setup
# # old_path = "Code4/DFEWdataset/Clip/clip_224x224" 
# # old_path = "F:/KmuProj2/Code4/DFEWdataset/Clip1/small_clip_224x224"
# # new_path = "F:/KmuProj2/Code4/DFEWdataset/Clip/small_clip_224x224"
# # old_path = "Code4/FERV39K/2_ClipsforFaceCrop"
# # new_path = "F:/KmuProj2/Code4/FERV39K/2_ClipsforFaceCrop"
# old_path = "Code4/AFEWdataset/AFEW_Face_Retina"
# new_path = "D:/KmuProj2/Code4/AFEWdataset/AFEW_Face_Retina"

# # Find all text files that match the pattern

# # all_txt_files = glob('DFEW_*.txt') 
# all_txt_files = glob('AFEW_*.txt')  # Ensure glob looks in the current directory
# for txt_file in all_txt_files:
#     update(txt_file, old_path, new_path)

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
old_path = "Code4/AFEWdataset/AFEW_Face_Retina"
new_path = "D:/KmuProj2/Code4/AFEWdataset/AFEW_Face_Retina"

# Find all text files (current folder). If needed, use recursive: glob('**/AFEW_*.txt', recursive=True)
all_txt_files = glob('AFEW_*.txt')
print("Found files:", all_txt_files)

for txt_file in all_txt_files:
    update(txt_file, old_path, new_path)
