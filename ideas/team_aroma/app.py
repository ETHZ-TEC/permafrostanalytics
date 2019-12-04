import streamlit as st
import pandas as pd
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

st.title("Data Exploration Engine")

st.write("Lets have a look at the data:")

@st.cache
def load_images(paths, color="gray"):
    image_files = []
    
    # Load just the first picture
    for f in paths[0:1]:
        image_files.append(sorted(glob.glob(f +"/*")))

    images = []
    if color == "gray":
        for ff in image_files:
            curr = []
            for f in ff:
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                curr.append(im)
            images.append(curr)
            
    else:
        for ff in image_files:
            curr = []
            for f in ff:
                im = cv2.imread(f, cv2.IMREAD_COLOR)
                curr.append(im)
            images.append(curr)
    return images


image_folders = sorted(glob.glob("../01_data/timelapse_images_fast/*"))
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
images_gray = load_images(image_folders)
images_col = load_images(image_folders, color="col")
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


hist_gray = cv2.calcHist([images_gray[0][0]],[0],None,[256],[0,256])
eq_hist_gray = cv2.calcHist([cv2.equalizeHist(images_gray[0][0])],[0],None,[256],[0,256])


### Visual Inspection
fig, ax = plt.subplots(3,2,figsize=(5,7))
ax[0,0].imshow(images_col[0][0])
ax[1,0].imshow(images_gray[0][0], cmap="gray")
color = ("b","g","r")
for i,col in enumerate(color):
    hist_col = cv2.calcHist(images_col[0][0],[i],None,[256],[0,256])
    ax[0,1].plot(hist_col,color=col)
ax[1,1].plot(hist_gray)
ax[2,0].imshow(images_gray[0][0], cmap="gray")
ax[2,1].plot(eq_hist_gray)
for row in np.arange(3):
    ax[row,0].yaxis.set_major_locator(plt.NullLocator())
    ax[row,0].yaxis.set_minor_locator(plt.NullLocator())
    ax[row,0].xaxis.set_major_locator(plt.NullLocator())
    ax[row,0].xaxis.set_minor_locator(plt.NullLocator())
    for i in ["top", "bottom", "right", "left"]:
        ax[row,0].spines[i].set_visible(False)
plt.show()
st.pyplot()