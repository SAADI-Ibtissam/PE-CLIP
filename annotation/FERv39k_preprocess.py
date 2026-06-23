import os
import pandas as pd
from glob import glob

FERV39K_SOURCE = "./FERV39K"
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
