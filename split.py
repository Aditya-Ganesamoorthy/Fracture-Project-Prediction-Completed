import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tqdm
import shutil

if __name__ == "__main__":

    print("🚀 Starting dataset split...")

    img_dir = "./GRAZPEDWRI-DX_dataset/data/images"
    ann_dir = "./GRAZPEDWRI-DX_dataset/data/labels"
    df = pd.read_csv("./GRAZPEDWRI-DX_dataset/dataset.csv")

    # First split: Train (70%) and Temp (30%)
    splitter1 = GroupShuffleSplit(test_size=0.3, n_splits=2, random_state=42)
    split = splitter1.split(df, groups=df["patient_id"])
    train_idxs, temp_idxs = next(split)

    train_df = df.iloc[train_idxs]
    temp_df = df.iloc[temp_idxs]

    # Second split: Validation (20%) and Test (10%)
    splitter2 = GroupShuffleSplit(test_size=0.33333, n_splits=2, random_state=42)
    split = splitter2.split(temp_df, groups=temp_df["patient_id"])
    valid_idxs, test_idxs = next(split)

    valid_df = temp_df.iloc[valid_idxs]
    test_df = temp_df.iloc[test_idxs]

    # Save split CSVs
    train_df.to_csv("./GRAZPEDWRI-DX_dataset/train_data.csv", index=False)
    valid_df.to_csv("./GRAZPEDWRI-DX_dataset/valid_data.csv", index=False)
    test_df.to_csv("./GRAZPEDWRI-DX_dataset/test_data.csv", index=False)

    # Create directories
    img_train_dir = "./GRAZPEDWRI-DX_dataset/data/images/train"
    img_valid_dir = "./GRAZPEDWRI-DX_dataset/data/images/valid"
    img_test_dir = "./GRAZPEDWRI-DX_dataset/data/images/test"

    ann_train_dir = "./GRAZPEDWRI-DX_dataset/data/labels/train"
    ann_valid_dir = "./GRAZPEDWRI-DX_dataset/data/labels/valid"
    ann_test_dir = "./GRAZPEDWRI-DX_dataset/data/labels/test"

    for dir_path in [img_train_dir, img_valid_dir, img_test_dir,
                     ann_train_dir, ann_valid_dir, ann_test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Function to move safely
    def move_files(df_subset, img_target_dir, ann_target_dir, split_name):
        missing_count = 0

        for i in tqdm.tqdm(df_subset.index, total=len(df_subset), desc=f"{split_name} split"):
            filestem = df_subset.loc[i, "filestem"]

            img_path = os.path.join(img_dir, filestem + ".png")
            label_path = os.path.join(ann_dir, filestem + ".txt")

            if os.path.exists(img_path) and os.path.exists(label_path):
                shutil.move(img_path, os.path.join(img_target_dir, filestem + ".png"))
                shutil.move(label_path, os.path.join(ann_target_dir, filestem + ".txt"))
            else:
                print(f"⚠️ Missing file: {filestem}")
                missing_count += 1

        print(f"✅ {split_name} split done. Missing files: {missing_count}")

    # Perform splitting
    move_files(train_df, img_train_dir, ann_train_dir, "Train")
    move_files(valid_df, img_valid_dir, ann_valid_dir, "Validation")
    move_files(test_df, img_test_dir, ann_test_dir, "Test")

    # Final stats
    N = len(df)
    print("\n📊 Data split completed according to PatientID:")
    print(f"  - {len(train_df)} ({100 * len(train_df)/N:.2f}%) → Train")
    print(f"  - {len(valid_df)} ({100 * len(valid_df)/N:.2f}%) → Validation")
    print(f"  - {len(test_df)} ({100 * len(test_df)/N:.2f}%) → Test")

    print("\n🎉 Dataset is ready for training!")