import kagglehub

# Download latest version of the dataset
path = kagglehub.dataset_download("lylmsc/wider-face-for-yolo-training")

print("✅ Download complete!")
print("📁 Dataset is available at:", path)
