import os
import pandas as pd
from glob import glob

DFEW_SOURCE = "Code4/DFEWdataset"
DFEW_CLIP_SOURCE = os.path.join(DFEW_SOURCE, "Clip", "small_clip_224x224")

def main():
    all_anno_file = sorted(glob(os.path.join(DFEW_SOURCE, "EmoLabel_DataSplit", "**", "*.csv")))
    print("ok1")
    for anno_file in all_anno_file:
        print(anno_file)
        data = pd.read_csv(anno_file)
        write_data = []
        write_file_path = "DFEW_{}_{}.txt".format(anno_file.split("/")[-1].split(".")[0], anno_file.split("/")[-2])
        write_file_path = os.path.join("Code4/DFEWdataset/Annotation", write_file_path)
        print("ok2")

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(write_file_path), exist_ok=True)

        for index, row in data.iterrows():
            video_name = str(row["video_name"]).zfill(5)
            full_video_name = os.path.join("Code4/DFEWdataset/Clip/clip_224x224", video_name)
            label = str(row["label"] - 1)
            clip_folder_path = os.path.join(DFEW_SOURCE, "Clip", "small_clip_224x224", video_name)
            
            # Debugging prints
            print(f"Processing video: {video_name}")
            print(f"Looking for images in: {clip_folder_path}")

            clip_image_path = sorted(glob(os.path.join(clip_folder_path, "*.jpg")))

            # Additional debugging prints
            print(f"Found {len(clip_image_path)} images in {clip_folder_path}")

            frame_num = len(clip_image_path)
            write_data.append("{} {} {}\n".format(full_video_name, frame_num, label))
        
        with open(write_file_path, "w") as f:
            f.writelines(write_data)

if __name__ == "__main__":
    main()
