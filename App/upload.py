from huggingface_hub import upload_folder

upload_folder(
    repo_id="drt-der/fraud_classification_reviews",
    folder_path="App/model",
    repo_type="model"
)