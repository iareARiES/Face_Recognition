import os
import shutil
from sklearn.model_selection import train_test_split

os.makedirs("dataset" , exist_ok = True)
base_dir  = "dataset"
image_dir = "WiderFace/images"
label_dir = "WiderFace/labels"



# Collect image and label and sort them
images = sorted([files for files in os.listdir(image_dir) if files.endswith(".jpg")])
labels = sorted([files for files in os.listdir(label_dir) if files.endswith(".txt")])
if len(images) == len(labels):
    print("Length Match")
else:
    raise ValueError(f"Mismatch: {len(images)} images and {len(labels)} labels!")
  
  
print(f"Length of images :{len(images)}")
print(f"Length of labels :{len(labels)}")


train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
def prepare_split(image_list , split):
  os.makedirs(f"{base_dir}/{split}/images", exist_ok=True)
  os.makedirs(f"{base_dir}/{split}/labels", exist_ok=True)
  
  for img in image_list:
    base = os.path.splitext(img)[0]
    shutil.copy(os.path.join(image_dir,img),f"{base_dir}/{split}/images/{img}")  #processing the images
    shutil.copy(os.path.join(label_dir,base+".txt"),f"{base_dir}/{split}/labels/{base}.txt") #processin the corresponding lables

#Executing the functions
prepare_split(train_images,"train")
prepare_split(val_images,"val")

print("Dataset has been split into train and validation sets.")

Setting up the yaml file
yaml_content = """\
path: dataset
train: train/images
val: val/images

nc: 1
names: ['face']
"""

with open("dataset/data.yaml", "w") as f:
    f.write(yaml_content)

print("YOLO data config file created at dataset/data.yaml")
!pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # You can use yolov8s.pt, yolov8m.pt, etc.

model.train(
    data="dataset/data.yaml",  # Path to the YAML file
    epochs=20,
    imgsz=640,
    batch=16
)

model = YOLO("runs/detect/train/weights/best.pt") #save this
