import zipfile
import os

print("Starting final zip file creation...")

base_dir = 'C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측'
zip_file_path = os.path.join(base_dir, 'submit', 'submit.zip')

with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add script.py
    script_path = os.path.join(base_dir, 'script.py')
    if os.path.exists(script_path):
        zipf.write(script_path, arcname='script.py')
        print("Added script.py")
    else:
        print("Error: script.py not found!")

    # Add requirements.txt
    req_path = os.path.join(base_dir, 'requirements.txt')
    if os.path.exists(req_path):
        zipf.write(req_path, arcname='requirements.txt')
        print("Added requirements.txt")
    else:
        print("Error: requirements.txt not found!")

    # Add model directory
    model_dir = os.path.join(base_dir, 'model')
    if os.path.isdir(model_dir):
        for root, _, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # The arcname preserves the 'model/...' structure
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname=arcname)
                print(f"Added {arcname}")
    else:
        print("Error: model/ directory not found!")

print(f"Successfully created {zip_file_path}")