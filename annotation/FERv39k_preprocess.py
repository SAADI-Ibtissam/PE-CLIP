# import os
# import pandas as pd
# from glob import glob

# FERV39K_SOURCE = "DFER-CLIP-main/FERV39K"
# CLIP_SOURCE = os.path.join(FERV39K_SOURCE, "2_ClipsforFaceCrop-001")

# def main():
#     all_anno_file = sorted(glob(os.path.join(FERV39K_SOURCE, "4_setups", "22_scenes", "*.csv")))
#     print("ok1")
    
#     for anno_file in all_anno_file:
#         print(anno_file)
#         data = pd.read_csv(anno_file)
#         write_data = []

#         is_train = "train" in os.path.basename(anno_file)
#         output_type = "train" if is_train else "test"
#         write_file_path = f"FERV39K_{output_type}.txt"

#         for index, row in data.iterrows():
#             video_name = str(row["video"]).zfill(4)  # Adjust the zfill value based on the actual length of video IDs
#             emotion_label = row["emotion"]
#             label = str(emotion_label - 1)  # Adjust the label to be zero-indexed
            
#             scene_name = os.path.basename(anno_file).split("_")[1].split(".")[0]
#             clip_folder_path = os.path.join(CLIP_SOURCE, scene_name, video_name)

#             # Debugging prints
#             print(f"Processing video: {video_name}")
#             print(f"Looking for images in: {clip_folder_path}")

#             clip_image_path = sorted(glob(os.path.join(clip_folder_path, "*.jpg")))

#             # Additional debugging prints
#             print(f"Found {len(clip_image_path)} images in {clip_folder_path}")

#             frame_num = len(clip_image_path)
#             full_video_name = os.path.join(CLIP_SOURCE, scene_name, video_name)
#             write_data.append(f"{full_video_name} {frame_num} {label}\n")
        
#         with open(write_file_path, "a") as f:  # Append mode to keep adding to the file
#             f.writelines(write_data)

# if __name__ == "__main__":
#     main()



import os
import pandas as pd
from glob import glob

FERV39K_SOURCE = "Code4/FERV39K"
CLIP_SOURCE = os.path.join(FERV39K_SOURCE, "2_ClipsforFaceCrop")

def process_annotations(annotation_file, is_train=True):
    data = pd.read_csv(annotation_file, header=None)  # Read the CSV without headers
    write_data = []

    emotion_mapping = {
        "Angry": 3,
        "Disgust": 5,
        "Fear": 6,
        "Happy": 0,
        "Neutral": 2,
        "Sad": 1,
        "Surprise": 4
    }

    for index, row in data.iterrows():
        # Split the single column into components
        entry = row[0]
        scene_emotion_video, emotion = entry.rsplit(' ', 1)

        # Further split the scene/emotion/video part
        scene, emotion_folder, video_id = scene_emotion_video.split('/')
        video_name = video_id.zfill(4)  # Adjust the zfill value based on the actual length of video IDs

        label = emotion_mapping[emotion]

        clip_folder_path = os.path.join(CLIP_SOURCE, scene, emotion_folder, video_name)
        
        # Debugging prints
        print(f"Processing video: {video_name}")
        print(f"Looking for images in: {clip_folder_path}")
        
        clip_image_path = sorted(glob(os.path.join(clip_folder_path, "*.jpg")))

        # Additional debugging prints
        print(f"Found {len(clip_image_path)} images in {clip_folder_path}")
        for img in clip_image_path:
            print(f"Image found: {img}")

        frame_num = len(clip_image_path)
        full_video_name = os.path.join(CLIP_SOURCE, scene, emotion_folder, video_name)
        write_data.append(f"{full_video_name} {frame_num} {label}\n")

    output_file = "FERV39K_train.txt" if is_train else "FERV39K_test.txt"
    with open(output_file, "a") as f:
        f.writelines(write_data)

def clear_output_files():
    with open("FERV39K_train.txt", "w") as f:
        f.write("")
    with open("FERV39K_test.txt", "w") as f:
        f.write("")

def main():
    # Clear previous output
    clear_output_files()

    train_files = sorted(glob(os.path.join(FERV39K_SOURCE, "4_setups", "22_scenes", "train_*.csv")))
    test_files = sorted(glob(os.path.join(FERV39K_SOURCE, "4_setups", "22_scenes", "test_*.csv")))

    for train_file in train_files:
        process_annotations(train_file, is_train=True)

    for test_file in test_files:
        process_annotations(test_file, is_train=False)

if __name__ == "__main__":
    main()




# import os
# import pandas as pd
# from glob import glob

# DFEW_SOURCE = "DFER-CLIP-main/FERV39K dataset"
# DFEW_CLIP_SOURCE = os.path.join(DFEW_SOURCE, "Clip", "clip_224x224")

# def main():
#     all_anno_file = sorted(glob(os.path.join(DFEW_SOURCE, "EmoLabel_DataSplit", "**", "*.csv")))
#     print("ok1")
#     for anno_file in all_anno_file:
#         print(anno_file)
#         data = pd.read_csv(anno_file)
#         write_data = []
#         write_file_path = "DFEW_{}_{}.txt".format(anno_file.split("/")[-1].split(".")[0], anno_file.split("/")[-2])
#         write_file_path = os.path.join("annotation", write_file_path)
#         print("ok2")

#         # Create the directory if it does not exist
#         os.makedirs(os.path.dirname(write_file_path), exist_ok=True)

#         for index, row in data.iterrows():
#             video_name = str(row["video_name"]).zfill(5)
#             full_video_name = os.path.join("DFER-CLIP-main/datasets/DFEW_Frame_Face", video_name)
#             label = str(row["label"] - 1)
#             clip_folder_path = os.path.join(DFEW_SOURCE, "Clip", "clip_224x224", video_name)
            
#             # Debugging prints
#             print(f"Processing video: {video_name}")
#             print(f"Looking for images in: {clip_folder_path}")

#             clip_image_path = sorted(glob(os.path.join(clip_folder_path, "*.jpg")))

#             # Additional debugging prints
#             print(f"Found {len(clip_image_path)} images in {clip_folder_path}")

#             frame_num = len(clip_image_path)
#             write_data.append("{} {} {}\n".format(full_video_name, frame_num, label))
        
#         with open(write_file_path, "w") as f:
#             f.writelines(write_data)

# if __name__ == "__main__":
#     main()
