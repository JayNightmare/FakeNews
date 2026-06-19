from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your dataset files
upload_folder(folder_path="src/datasets/data/master_dataset", repo_id="JayNightmare/Misinformation-Dataset", repo_type="dataset")
