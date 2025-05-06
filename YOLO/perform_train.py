#%% 
from ultralytics import YOLO

# sources: 
# https://docs.ultralytics.com/cli/
# https://docs.ultralytics.com/cfg/
# %% load the model
#model = YOLO("yolov11n.yaml")  # build a new model from scratch
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# %% Train the model
results = model.train(data="train_custom/masks.yaml", epochs=50, imgsz=512, batch=4, verbose=True, device='cpu')
# device='cuda' to use GPU
# print(model.device) # to check/confirm
# %% Export the model
model.export()
# .. and got:
#TorchScript: starting export with torch 2.2.2...
#TorchScript: export success ✅ 4.3s, saved as 'runs/detect/train6/weights/best.torchscript' (10.4 MB)