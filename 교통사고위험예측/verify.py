import zipfile
import os

print("Starting verification...")

base_dir = 'C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측'
zip_file_path = os.path.join(base_dir, 'submit', 'submit.zip')
verify_dir = os.path.join(base_dir, 'temp_verify')

try:
    # 1. Create verification directory
    os.makedirs(verify_dir, exist_ok=True)
    print(f"Created directory: {verify_dir}")

    # 2. Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(verify_dir)
    print(f"Unzipped {zip_file_path} to {verify_dir}")

    # 3. List the contents
    unzipped_files = os.listdir(verify_dir)
    print(f"Contents of temp_verify: {unzipped_files}")

except Exception as e:
    print(f"An error occurred during verification: {e}")
    exit(1)

print("Verification script finished.")
