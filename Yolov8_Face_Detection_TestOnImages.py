from IPython.display import Image, display

display(Image(filename="runs/detect/train/results.png"))
#Running inference on the images.
results = model.predict(source="dataset/val/images", save=True, imgsz=640)
from IPython.display import Image, display
display(Image(filename='runs/detect/predict/wider_261.jpg'))  # or any image in the folder
