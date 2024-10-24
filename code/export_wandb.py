import wandb
import os
import zipfile

api = wandb.Api()

run_path = input("Enter the run path (entity/project/run_id): ")

run = api.run(run_path)

run_dir = input("Enter the folder name to save the run files: ")
os.makedirs(run_dir, exist_ok=True)

for file in run.files():
    file.download(run_dir)

zip_filename = input("Enter the name for the zip file (e.g., wandb_run.zip): ")
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            zipf.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                       os.path.join(run_dir, '..')))

print(f"Run files have been exported to {zip_filename}")
