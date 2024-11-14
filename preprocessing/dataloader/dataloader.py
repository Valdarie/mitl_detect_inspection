import subprocess
import os

def run_image_slicing():
    python_executable = os.path.join(".venv", "Scripts", "python.exe")
    coco_file_name = 'cassette1'
    data_dir = "data/coco_json_files"
    splits = ['train', 'val', 'test']

    split_script = os.path.join("preprocessing", "traintestval_split.py")
    slicing_script = os.path.join("preprocessing", "image_slicing.py")

    # Step 1: Run train_testval_split.py to split the data
    print(f"Running {split_script}...")
    subprocess.run([python_executable, split_script])
    print(f"Completed {split_script}.")

    # Step 2: Run image_slicing.py for each split
    split_files = []
    for split in splits:
        # Define the input JSON file path for each split
        split_file = os.path.join(data_dir, f"{coco_file_name}_{split}.json")
        split_files.append(split_file)
        
        print(f"Running {slicing_script} on {split_file} for {split} split...")
        # Pass only annotation_file and split arguments to image_slicing.py
        subprocess.run([python_executable, slicing_script, split_file, split])
        print(f"Completed slicing for {split_file}.")

    print("Data loading and slicing process completed for all splits.")
    
    return split_files

# Calling the function and capturing the returned split files
#if __name__ == "__main__":
    #split_files = run_image_slicing()
    #print(f"Split files: {split_files}")