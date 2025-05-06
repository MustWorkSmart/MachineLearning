#%% package import
import pandas as pd
import numpy as np
#!pip install seaborn #as needed
import seaborn as sns
import os
#!pip install shutil #as needed
import shutil
import xml.etree.ElementTree as ET
import glob

import json

#for MacOS
MAC = True
#%% Function for conversion XML to YOLO
# based on https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5
def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

#%% create folders
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
if MAC:
    #for Mac, need to use these instead:
    create_folder(r'yolov11/train_custom/train/images')
    create_folder(r'yolov11/train_custom/train/labels')
    create_folder(r'yolov11/train_custom/val/images')
    create_folder(r'yolov11/train_custom/val/labels')
    create_folder(r'yolov11/train_custom/test/images')
    create_folder(r'yolov11/train_custom/test/labels')
else:
    create_folder('yolov11\\train_custom\\train\\images')
    create_folder('yolov11\\train_custom\\train\\labels')
    create_folder('yolov11\\train_custom\\val\\images')
    create_folder('yolov11\\train_custom\\val\\labels')
    create_folder('yolov11\\train_custom\\test\\images')
    create_folder('yolov11\\train_custom\\test\\labels')

#no need to try this for now:
#path = os.path.join('yolov7', 'train', 'images')
#create_folder(path)

#%% get all image files
img_folder = 'images'
_, _, files = next(os.walk(img_folder))
pos = 0
for f in files:
        source_img = os.path.join(img_folder, f)
        if pos < 700: #future to-do: will need to select randomly instead. also ensure balanced # of labels distribution
            dest_folder = 'yolov11/train_custom/train' if MAC else 'yolov11\\train_custom\\train'
        elif (pos >= 700 and pos < 800):
            dest_folder = 'yolov11/train_custom/val' if MAC else 'yolov11\\train_custom\\val'
        else:
            dest_folder = 'yolov11/train_custom/test' if MAC else 'yolov11\\train_custom\\test'
        destination_img = os.path.join(dest_folder,'images', f)
        shutil.copy(source_img, destination_img)

        # check for corresponding label
        label_file_basename = os.path.splitext(f)[0]
        label_source_file = f"{label_file_basename}.xml"
        label_dest_file = f"{label_file_basename}.txt"
        
        label_source_path = os.path.join('annotations', label_source_file)
        label_dest_path = os.path.join(dest_folder, 'labels', label_dest_file)
        # if file exists, copy it to target folder
        if os.path.exists(label_source_path):
             # parse the content of the xml file
            tree = ET.parse(label_source_path)
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
            result = []
            for obj in root.findall('object'):
                label = obj.find("name").text
                # check for new classes and append to list
                index = classes.index(label)
                pil_bbox = [int(x.text) for x in obj.find("bndbox")]
                yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                # convert data to string
                bbox_string = " ".join([str(x) for x in yolo_bbox])
                result.append(f"{index} {bbox_string}")
                if result:
                    # generate a YOLO format text file for each xml file
                    with open(label_dest_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(result))                        
        pos += 1        
# %%
#!pip install pytube #if needed
#from pytube import YouTube
#YouTube('https://youtu.be/AYCekXizlf0').streams.first().download()
#yt = YouTube('http://youtube.com/watch?v=2lAe1cqCOXo')
#yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()

#below got: HTTPError: HTTP Error 400: Bad Request
'''
def Download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")

#link = input("Enter the YouTube video URL: ")
link = r"https://www.youtube.com/watch?v=AYCekXizlf0"
print(link)
Download(link)
'''

#https://github.com/pytube/pytube/issues/2044
#!pip install pytubefix #if needed
from pytubefix import YouTube
import time

# getting video for testing later on
video_url = 'https://www.youtube.com/watch?v=AYCekXizlf0'

try:
    # Create a YouTube object with additional configuration
    yt = YouTube(
        video_url
    )

    # Get the lowest resolution stream
    video_stream = yt.streams.get_lowest_resolution()
    
    # Download the video
    print(f"Downloading: {yt.title}")
    video_stream.download(output_path=".")
    print("Download complete!")
except Exception as e:
    print(f"An error occurred: {e}")

#move the downloaded video to: yolo11/train_custom/test/images/.
import os
current_directory = os.getcwd()
print("current directory is:", current_directory)
entries = os.listdir(current_directory)
videos = [] #for now, there would be only 1 video, based on codes above
for entry in entries:
    print(entry)
    if os.path.isfile(entry) and os.access(entry, os.R_OK) and entry.lower().endswith(".mp4"):
        videos.append(entry)
print(f"got these videos: {videos}")
destination_path = r"yolov11/train_custom/test/images/" #videos there too
for video in videos:
    shutil.copy(video, destination_path)
print(f"Finished copying {len(videos)} videos to: ", destination_path)

# %%
