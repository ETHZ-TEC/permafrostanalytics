import stuett
from stuett.global_config import get_setting, setting_exists
import pandas as pd
from skimage import io as imio
import io
import cv2
import sys
import numpy as np
from functools import lru_cache
import pickle

def wait():
    while True:
        key = cv2.waitKey()
        print(key)
        if key == 13:
            break

account_name = (
get_setting("azure")["account_name"]
if setting_exists("azure")
else "storageaccountperma8980"
)
account_key = (
get_setting("azure")["account_key"] if setting_exists("azure") else None
)
prefix = "timelapse_images"
store = stuett.ABSStore(
container="hackathon-on-permafrost",
prefix=prefix,
account_name=account_name,
account_key=account_key, 
)

account_name = "storageaccountperma8980"
account_key = None


data = stuett.data.MHDSLRFilenames(
    store=store,
    start_time=pd.to_datetime("2017-01-01"),
    end_time=pd.to_datetime("2017-12-31"),
)()


rows = data.shape[0]

index=0
index2 = rows-1

@lru_cache(maxsize=32)
def read_image(idx):
    key = data.iloc[idx]["filename"]
    img = imio.imread(io.BytesIO(store[key]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def align_images(ref, img):
    # Read the images to be aligned
    im1 = ref
    im2 = img
     
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
     
    # Find size of image1
    sz = im1.shape
     
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
     
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
     
    # Specify the number of iterations.
    number_of_iterations = 5000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None, 5)
     
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    return im2_aligned

al_cache = {}
def get_align_cache(idx_ref, idx_img):
    if not (idx_ref, idx_img) in al_cache.keys():
        img = read_image(idx_img)
        return img
    return al_cache[(idx_ref, idx_img)]

def align_cache(idx_ref, idx_img):
    if not (idx_ref, idx_img) in al_cache.keys():
        ref = read_image(idx_ref)
        img = read_image(idx_img)
        al_cache[(idx_ref, idx_img)] = align_images(ref, img)
    return al_cache[(idx_ref, idx_img)]

def get_date(idx):
    return data.index[idx]

def get_next_day_idx(idx):
    return data.index.get_loc(data.index[idx] + pd.DateOffset(days=1), method='nearest')
def get_prev_day_idx(idx):
    return data.index.get_loc(data.index[idx] - pd.DateOffset(days=1), method='nearest')


ix,iy = -1,-1
# mouse callback function
def mouse(event,x,y,flags,param):
    global ix,iy
    ix,iy = x,y


front = True

cv2.namedWindow('a',cv2.WINDOW_NORMAL)
cv2.resizeWindow('a', 600,600)
cv2.setMouseCallback('a',mouse)

#[((x, y), (first image, last image)), ...]
found_positions = []
selected_position = 0
try:
    found_positions = pickle.load( open( "pos.p", "rb" ) )
except Exception as e:
    print(e)

nav_queue = []



while True:
    info_img = np.zeros((512,1024,3), np.uint8)

    idx = index if front else index2

    front_img = read_image(index)
    back_img_al = get_align_cache(index, index2)

    img = front_img if front else back_img_al
    img = img.copy() # So we dont draw into cache

    i = 0
    y = 90
    for p, idxs in found_positions:
        if i == selected_position:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.circle(img, p, 10, color, 2)
        cv2.circle(img, p, 100, color, 5)
        i += 1
        cv2.putText(info_img,'Pos: {} - {}'.format(p, idxs), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y += 30

    cv2.putText(info_img,'Front: {} - {}'.format(index, get_date(index)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if front else (0, 0, 255), 2)
    cv2.putText(info_img,'Back:  {} - {}'.format(index2, get_date(index2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if front else (0, 255, 0), 2)

    
    cv2.imshow('a', img)
    cv2.imshow('info', info_img)

    while True:
        key = cv2.waitKey()
        if key == 113:
            pickle.dump( found_positions, open( "pos.p", "wb" ) )
            sys.exit(0)
        if key == ord('s'):
            front = not front
            break
        if key == ord('j'): 
            if front:
                index = max(0, index - 1)
            else:
                index2 = max(0, index2 - 1)
            break
        if key == ord('k'):
            if front:
                index = min(rows-1, index + 1)
            else:
                index2 = min(rows-1, index2 + 1)
            break
        if key == ord('l'): 
            if front:
                index = get_next_day_idx(index)
            else:
                index2 = get_next_day_idx(index2)
            break
        if key == ord('h'): 
            if front:
                index = get_prev_day_idx(index)
            else:
                index2 = get_prev_day_idx(index2)
            break
        if key == ord('a'):
            align_cache(index, index2)
            break
        if key == ord('f'):
            found_positions.append(((ix, iy), (index, index2)))
            break
        if key == ord('m'):
            #Jump to middle of time range
            nav_queue.append((index, index2))
            if front:
                index = int((index + index2) / 2)
            else:
                index2 = int((index + index2) / 2)
            break
        if key == ord('b'):
            #Jump back to range before split
            if len(nav_queue) > 0:
                index, index2 = nav_queue.pop()
            break
        if key == ord('v'):
            #Switch with neighboring range
            if len(nav_queue) > 0:
                last_middle = int((nav_queue[-1][0] + nav_queue[-1][1]) / 2)
                if abs(index2 - last_middle) < abs(index - last_middle):
                    index = index2
                    index2 = nav_queue[-1][1]
                else:
                    index2 = index
                    index = nav_queue[-1][0]
            break
        if key == ord('i'):
            selected_position = max(0, selected_position - 1)
            break
        if key == ord('o'):
            selected_position = min(len(found_positions) - 1, selected_position + 1)
            break
        if key == ord('p'):
            # Jump to position range
            nav_queue.append((index, index2))
            index, index2 = found_positions[selected_position][1]
            break
        if key == ord('c'):
            # Set to current range
            pos, ran = found_positions[selected_position]
            found_positions[selected_position] = (pos, (index, index2))
            break
        if key == ord('r'):
            # Delete completely
            del found_positions[selected_position]
            break
