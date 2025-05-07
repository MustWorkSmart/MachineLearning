#%%
#!pip install ultralytics #as needed
from ultralytics import YOLO
# %% Load a model
#model = YOLO("yolo11n.pt") # load our custom trained model
model = YOLO("runs/detect/train6/weights/best.pt") # load our custom trained model

# %%
#result = model("test/corgi_1.jpg")
# %%
#result
# %% command line run
# Standard Yolo
!yolo detect predict model=yolo11n.pt source="test/corgi_1.jpg" conf=0.3 
# %% Masks 
!yolo detect predict model=yolo11n.pt source="test/DJT.jpg" conf=0.3 
!yolo detect predict model=yolo11n.pt source="test/maksssksksss10.png" conf=0.3 
!yolo detect predict model=yolo11n.pt source="test/Mask wearing.mp4" conf=0.3 

# %%
!yolo detect predict model="runs/detect/train6/weights/best.pt" source="test/corgi_1.jpg" conf=0.3 
!yolo detect predict model="runs/detect/train6/weights/best.pt" source="test/DJT.jpg" conf=0.3 
!yolo detect predict model="runs/detect/train6/weights/best.pt" source="test/maksssksksss10.png" conf=0.3 
!yolo detect predict model="runs/detect/train6/weights/best.pt" source="test/Mask wearing.mp4" conf=0.3 

# %%
